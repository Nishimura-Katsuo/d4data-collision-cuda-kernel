#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kFieldMask = 0x0FFFFFFFu;
constexpr uint64_t kEmptyTableEntry = 0ull;
constexpr uint32_t kDefaultMinLength = 1;
constexpr uint32_t kDefaultMaxLength = 64;
constexpr uint64_t kDefaultBatchSize = 1ull << 22;

struct SearchCount {
    uint64_t low = 0;
    uint64_t high = 0;

    SearchCount& operator+=(uint64_t value) {
        const uint64_t previousLow = low;
        low += value;
        high += (low < previousLow) ? 1u : 0u;
        return *this;
    }

    SearchCount& operator+=(const SearchCount& other) {
        const uint64_t previousLow = low;
        low += other.low;
        high += other.high;
        high += (low < previousLow) ? 1u : 0u;
        return *this;
    }
};
constexpr uint32_t kMatchBufferCapacity = 1u << 18;

std::atomic<bool> g_interrupted{false};

void handleSignal(int) {
    g_interrupted.store(true, std::memory_order_relaxed);
}

enum class Mode {
    Type,
    Field,
    Gbid,
};

struct Options {
    Mode mode = Mode::Type;
    bool useEnglish = false;
    bool useExpanded = false;
    bool noDict = false;
    bool literal = false;
    bool wordsOnly = false;
    bool allowAllCaps = false;
    bool noPrefix = false;
    bool useCommonPrefixes = true;
    bool force = false;
    bool paired = false;
    bool logMatches = false;
    uint32_t minLength = kDefaultMinLength;
    uint32_t maxLength = kDefaultMaxLength;
    bool minSpecified = false;
    bool maxSpecified = false;
    uint32_t threads = 1;
    std::optional<std::string> dictArg;
    std::unordered_map<size_t, std::vector<std::string>> positionTokens;
    std::optional<std::vector<std::string>> suffixTokens;
    std::vector<uint32_t> explicitTargets;
};

struct Token {
    std::string text;
    uint32_t hash = 0;
    uint16_t length = 0;
};

struct DeviceToken {
    uint16_t length;
    uint16_t reserved;
    uint32_t hash;
};

struct DeviceMatch {
    uint32_t hash;
    uint64_t suffixIndex;
};

struct DictionaryBuild {
    std::vector<uint32_t> mainPool;
    bool dictionaryEnabled = false;
};

struct LengthPlan {
    uint32_t length = 0;
    std::vector<std::vector<uint32_t>> pools;
    std::vector<size_t> poolSizes;
    size_t gpuStartPosition = 0;
    uint64_t suffixSearchSpace = 0;
};

struct SearchContext {
    Options options;
    std::vector<Token> tokens;
    std::unordered_map<std::string, uint32_t> tokenIds;
    std::vector<uint32_t> pow33;
    std::vector<uint32_t> mainPool;
    std::vector<uint32_t> targets;
    std::unordered_set<uint32_t> targetLookup;
    std::unordered_set<uint32_t> relevantUnfoundLookup;
    std::unordered_set<uint32_t> unfoundFieldLookup;
    std::unordered_map<uint32_t, std::vector<uint32_t>> fieldToTypes;
    std::unordered_map<uint32_t, std::unordered_set<std::string>> typePrefixes;
    std::unordered_map<uint32_t, uint32_t> prefixOrdinalByTokenId;
    std::unordered_map<uint32_t, uint32_t> targetIndexByHash;
    std::vector<uint64_t> targetAllowedFirstTokenMasks;
    bool usingFieldTypeMap = false;
    bool canUseGpuFieldTypeGate = false;
    bool checksumProvided = false;
    size_t typePrefixSize = 0;
};

struct GpuBuffers {
    struct BatchSlot {
        DeviceMatch* d_matches = nullptr;
        uint32_t* d_matchCount = nullptr;
        uint32_t* d_overflow = nullptr;
        DeviceMatch* hostMatches = nullptr;
        uint32_t* hostMatchCount = nullptr;
        uint32_t* hostOverflow = nullptr;
        cudaStream_t stream = nullptr;
    };

    uint32_t* d_pow33 = nullptr;
    uint64_t* d_targetTable = nullptr;
    uint64_t* d_targetPrefixMasks = nullptr;
    uint32_t* d_poolOffsets = nullptr;
    uint32_t* d_poolSizes = nullptr;
    DeviceToken* d_poolTokens = nullptr;
    uint32_t targetMask = 0;
    bool fieldGateEnabled = false;
    size_t poolOffsetsCapacity = 0;
    size_t poolSizesCapacity = 0;
    size_t poolTokensCapacity = 0;
    int blockSize = 256;
    int gridSize = 1;
    std::array<BatchSlot, 2> slots;
};

struct OutputSink {
    explicit OutputSink(bool enabled, Mode mode) : enabled(enabled), mode(mode) {
        if (enabled) {
            const fs::path directory = outputDirectory();
            fs::create_directories(directory);
        }
    }

    void emit(uint32_t hash, const std::string& candidate) {
        const std::string hex = toHex(hash);
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "  " << hex << ": " << candidate << '\n' << std::flush;
        if (!enabled) {
            return;
        }
        const fs::path directory = outputDirectory();
        std::ofstream output(directory / (hex + ".yml"), std::ios::app);
        output << hex << ": " << candidate << '\n';
    }

    fs::path outputDirectory() const {
        if (mode == Mode::Field) {
            return fs::path("field");
        }
        if (mode == Mode::Gbid) {
            return fs::path("gbid");
        }
        return fs::path("type");
    }

    static std::string toHex(uint32_t value) {
        std::ostringstream stream;
        stream << std::hex << std::nouppercase << value;
        return stream.str();
    }

    bool enabled;
    Mode mode;
    std::mutex mutex;
};

uint32_t mixHash(uint32_t current, unsigned char byte) {
    return current * 33u + static_cast<uint32_t>(byte);
}

uint32_t hashString(std::string_view text) {
    uint32_t hash = 0;
    for (unsigned char byte : text) {
        hash = mixHash(hash, byte);
    }
    return hash;
}

uint32_t hashStringLower(std::string_view text) {
    uint32_t hash = 0;
    for (unsigned char byte : text) {
        hash = mixHash(hash, static_cast<unsigned char>(std::tolower(byte)));
    }
    return hash;
}

uint32_t hashForMode(Mode mode, std::string_view text) {
    return mode == Mode::Gbid ? hashStringLower(text) : hashString(text);
}

__host__ __device__ uint32_t combineHash(uint32_t prefixHash, uint32_t tokenHash, uint32_t pow33ForLength) {
    return prefixHash * pow33ForLength + tokenHash;
}

std::string stripCarriageReturn(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

bool hasNoLowercaseAscii(const std::string& value) {
    if (value.size() < 2) {
        return false;
    }
    for (unsigned char byte : value) {
        if (byte >= 'a' && byte <= 'z') {
            return false;
        }
    }
    return true;
}

std::string lowercaseAsciiCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char byte) {
        return static_cast<char>(std::tolower(byte));
    });
    return value;
}

std::optional<std::string> normalizeDictionaryToken(const std::string& raw, const Options& options) {
    if (raw.empty()) {
        return std::nullopt;
    }
    if (options.literal) {
        return raw;
    }
    if (raw.size() < 2 && !options.wordsOnly) {
        return std::nullopt;
    }
    if (hasNoLowercaseAscii(raw) && !options.allowAllCaps) {
        return std::nullopt;
    }
    if (options.mode == Mode::Gbid) {
        return lowercaseAsciiCopy(raw);
    }
    const std::string lowered = lowercaseAsciiCopy(raw);
    if (raw == lowered) {
        std::string normalized = lowered;
        normalized[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized[0])));
        return normalized;
    }
    return raw;
}

std::vector<std::string> splitWhitespace(const std::string& value) {
    std::vector<std::string> tokens;
    std::istringstream stream(value);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> parseTokenArgument(const std::string& value) {
    if (value.empty()) {
        return {""};
    }
    return splitWhitespace(value);
}

std::string lowercaseFirstCharacter(const std::string& value) {
    if (value.empty()) {
        return value;
    }
    std::string lowered = value;
    lowered[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(lowered[0])));
    return lowered;
}

std::string trimHexPrefix(std::string value) {
    if (value.size() >= 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
        return value.substr(2);
    }
    return value;
}

uint32_t parseHexValue(const std::string& value) {
    const std::string stripped = trimHexPrefix(value);
    size_t consumed = 0;
    const unsigned long parsed = std::stoul(stripped, &consumed, 16);
    if (consumed != stripped.size()) {
        throw std::runtime_error("invalid hex value: " + value);
    }
    return static_cast<uint32_t>(parsed);
}

std::vector<uint32_t> loadHexList(const fs::path& path) {
    std::vector<uint32_t> values;
    std::ifstream input(path);
    if (!input) {
        return values;
    }
    std::string line;
    while (std::getline(input, line)) {
        line = stripCarriageReturn(std::move(line));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        values.push_back(parseHexValue(line));
    }
    return values;
}

uint32_t getOrCreateToken(SearchContext& context, const std::string& text) {
    const auto it = context.tokenIds.find(text);
    if (it != context.tokenIds.end()) {
        return it->second;
    }
    Token token;
    token.text = text;
    token.hash = hashForMode(context.options.mode, text);
    token.length = static_cast<uint16_t>(text.size());
    const uint32_t id = static_cast<uint32_t>(context.tokens.size());
    context.tokens.push_back(std::move(token));
    context.tokenIds.emplace(text, id);
    return id;
}

void appendTokenIfMissing(SearchContext& context,
                         std::vector<uint32_t>& pool,
                         std::unordered_set<std::string>* seen,
                         const std::string& text) {
    if (seen != nullptr && !seen->emplace(text).second) {
        return;
    }
    pool.push_back(getOrCreateToken(context, text));
}

void appendBuiltins(SearchContext& context,
                    std::vector<uint32_t>& pool,
                    std::unordered_set<std::string>* seen) {
    if (context.options.mode != Mode::Gbid) {
        for (char value = 'A'; value <= 'Z'; ++value) {
            appendTokenIfMissing(context, pool, seen, std::string(1, value));
        }
    }
    for (char value = 'a'; value <= 'z'; ++value) {
        appendTokenIfMissing(context, pool, seen, std::string(1, value));
    }
    for (char value = '0'; value <= '9'; ++value) {
        appendTokenIfMissing(context, pool, seen, std::string(1, value));
    }
    appendTokenIfMissing(context, pool, seen, "_");
}

void appendNormalizedTokens(SearchContext& context,
                            std::vector<uint32_t>& destination,
                            const std::vector<std::string>& source,
                            std::unordered_set<std::string>* seen,
                            bool deduplicateWithinSource) {
    std::unordered_set<std::string> localSeen;
    for (const std::string& token : source) {
        const std::optional<std::string> normalized = normalizeDictionaryToken(token, context.options);
        if (!normalized.has_value()) {
            continue;
        }
        if (deduplicateWithinSource && !localSeen.emplace(*normalized).second) {
            continue;
        }
        appendTokenIfMissing(context, destination, seen, *normalized);
    }
}

std::vector<std::string> loadDictionaryLines(const fs::path& path, bool* openedSuccessfully) {
    std::vector<std::string> lines;
    std::ifstream input(path);
    if (!input) {
        *openedSuccessfully = false;
        return lines;
    }
    *openedSuccessfully = true;
    std::string line;
    while (std::getline(input, line)) {
        line = stripCarriageReturn(std::move(line));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        lines.push_back(line);
    }
    return lines;
}

DictionaryBuild buildMainPool(SearchContext& context) {
    DictionaryBuild build;
    std::unordered_set<std::string> seen;
    if (!context.options.noDict) {
        build.dictionaryEnabled = true;
        auto appendDictionaryFile = [&](const fs::path& path) {
            bool loadedFromFile = false;
            const std::vector<std::string> rawTokens = loadDictionaryLines(path, &loadedFromFile);
            if (!loadedFromFile) {
                std::cerr << "Dictionary " << path.string() << " not found." << '\n';
                return;
            }
            appendNormalizedTokens(context, build.mainPool, rawTokens, &seen, true);
        };

        if (context.options.dictArg.has_value()) {
            if (*context.options.dictArg == "english_dict.txt") {
                appendDictionaryFile("dict.txt");
            }
            appendDictionaryFile(*context.options.dictArg);
        } else {
            if (context.options.useEnglish) {
                appendDictionaryFile("dict.txt");
                appendDictionaryFile("english_dict.txt");
            } else if (context.options.useExpanded) {
                appendDictionaryFile("dict_expanded.txt");
            } else {
                appendDictionaryFile("dict.txt");
            }
        }
    }
    if (!(context.options.wordsOnly && build.dictionaryEnabled)) {
        appendBuiltins(context, build.mainPool, &seen);
    }
    return build;
}

void addTypePrefixes(std::unordered_map<uint32_t, std::unordered_set<std::string>>* typePrefixes,
                     uint32_t typeHash,
                     std::initializer_list<const char*> common,
                     std::initializer_list<const char*> uncommon,
                     bool useCommonOnly) {
    std::vector<std::string> additions;
    for (const char* prefix : common) {
        additions.emplace_back(prefix);
    }
    if (!useCommonOnly) {
        for (const char* prefix : uncommon) {
            additions.emplace_back(prefix);
        }
    }
    if (additions.empty()) {
        return;
    }
    auto& prefixes = (*typePrefixes)[typeHash];
    for (const std::string& prefix : additions) {
        prefixes.insert(prefix);
    }
}

std::unordered_map<uint32_t, std::unordered_set<std::string>> buildTypePrefixes(bool useCommonOnly) {
    std::unordered_map<uint32_t, std::unordered_set<std::string>> typePrefixes;
    typePrefixes[0].insert("t");

    const uint32_t hashBool = hashString("DT_BOOL");
    const uint32_t hashFloatArray = hashString("DT_FLOAT_ARRAY");
    const uint32_t hashAabb = hashString("AABB");
    const uint32_t hashAxialCylinder = hashString("AxialCylinder");
    const uint32_t hashAcdNetworkName = hashString("DT_ACD_NETWORK_NAME");
    const uint32_t hashBcVec2i = hashString("DT_BCVEC2I");
    const uint32_t hashByte = hashString("DT_BYTE");
    const uint32_t hashCharArray = hashString("DT_CHARARRAY");
    const uint32_t hashCString = hashString("DT_CSTRING");
    const uint32_t hashEnum = hashString("DT_ENUM");
    const uint32_t hashFixedArray = hashString("DT_FIXEDARRAY");
    const uint32_t hashFloat = hashString("DT_FLOAT");
    const uint32_t hashGbid = hashString("DT_GBID");
    const uint32_t hashInt = hashString("DT_INT");
    const uint32_t hashInt64 = hashString("DT_INT64");
    const uint32_t hashPolyArray = hashString("DT_POLYMORPHIC_VARIABLEARRAY");
    const uint32_t hashRange = hashString("DT_RANGE");
    const uint32_t hashRgbaColor = hashString("DT_RGBACOLOR");
    const uint32_t hashRgbaColorValue = hashString("DT_RGBACOLORVALUE");
    const uint32_t hashSharedServerDataId = hashString("DT_SHARED_SERVER_DATA_ID");
    const uint32_t hashSno = hashString("DT_SNO");
    const uint32_t hashSnoName = hashString("DT_SNO_NAME");
    const uint32_t hashStartlocName = hashString("DT_STARTLOC_NAME");
    const uint32_t hashStringFormula = hashString("DT_STRING_FORMULA");
    const uint32_t hashTagMap = hashString("DT_TAGMAP");
    const uint32_t hashUint = hashString("DT_UINT");
    const uint32_t hashVariableArray = hashString("DT_VARIABLEARRAY");
    const uint32_t hashVector2d = hashString("DT_VECTOR2D");
    const uint32_t hashVector3d = hashString("DT_VECTOR3D");
    const uint32_t hashVector4d = hashString("DT_VECTOR4D");
    const uint32_t hashWord = hashString("DT_WORD");
    const uint32_t hashGbHandle = hashString("GBHandle");
    const uint32_t hashInterpolationRgba = hashString("InterpolationPath_RGBAColor");
    const uint32_t hashInterpolationFloat = hashString("InterpolationPath_float");
    const uint32_t hashInterpolationInt32 = hashString("InterpolationPath_int32");
    const uint32_t hashMatrix3x3 = hashString("Matrix3x3");
    const uint32_t hashPrsTransform = hashString("PRSTransform");
    const uint32_t hashPrTransform = hashString("PRTransform");
    const uint32_t hashSharedServerWorldPlace = hashString("SharedServerWorldPlace");
    const uint32_t hashSpeedTreeBranchWindLevel = hashString("SpeedTree8BranchWindLevel");
    const uint32_t hashSpeedTreeRippleGroup = hashString("SpeedTree8RippleGroup");
    const uint32_t hashSphere = hashString("Sphere");
    const uint32_t hashStringLabelHandleEx = hashString("StringLabelHandleEx");
    const uint32_t hashUiControlHandle = hashString("UIControlHandle");
    const uint32_t hashUiImageHandleReference = hashString("UIImageHandleReference");
    const uint32_t hashVectorPath = hashString("VectorPath");
    const uint32_t hashBcQuat = hashString("bcQuat");
    const uint32_t hashDmTransformMirror = hashString("dmTransformMirror");
    const uint32_t hashMystery = 0xf5ac91bbu;

    addTypePrefixes(&typePrefixes, hashBool, {"b", "f"}, {"m_b", "m_f"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashFloatArray, {"af"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashAabb, {"aabb"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashGbHandle, {}, {"h"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSharedServerWorldPlace, {"wp"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashStringLabelHandleEx, {"h"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashByte, {"dw"}, {"u", "n", "game", "twin"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashEnum, {"e"}, {"id", "n", "dw"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashGbid, {"gbid"}, {"n"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashUint, {"dw", "h", "n", "sz", "u"}, {"s", "id", "w", "sno"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashWord, {"bone", "dw", "n", "u"}, {"vertex", "triangle", "constraint", "max", "plane", "m_bone", "attachment", "start", "end"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashInterpolationRgba, {"path"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashDmTransformMirror, {"local", "m_local"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashPolyArray, {"ar", "pt"}, {"arn", "arr", "at"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashUiImageHandleReference, {"h"}, {"ar"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashBcVec2i, {"vec", "p"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashMatrix3x3, {"m"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashCharArray, {"sz", "us"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashStartlocName, {"u", "dw"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashRgbaColor, {"rgba"}, {"fl"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashFixedArray, {"ar", "pt"}, {"arn", "arr", "at"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashStringFormula, {"v", "sz"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashPrTransform, {"local"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashVectorPath, {"path"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashInt, {"dw", "n"}, {"count", "e", "i", "id", "is", "m_", "m_bone", "m_face", "m_plane", "m_vertex", "sample", "sno", "w"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSno, {"sno"}, {"h"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashAcdNetworkName, {"ann"}, {"m_ann"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashAxialCylinder, {"wcyl"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSharedServerDataId, {"id"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashUiControlHandle, {"h"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashVector2d, {"v", "vec"}, {"wp", "wv"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashVector3d, {"wv", "wp", "v", "vec"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashVector4d, {"v"}, {"inv", "m_inv", "pt", "vec"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashRgbaColorValue, {"rgbaval"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashVariableArray, {"ar", "pt"}, {"arn", "arr", "at"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSnoName, {"snoname", "sno"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSphere, {"ws"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashInterpolationFloat, {"path"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSpeedTreeBranchWindLevel, {"s"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashTagMap, {"m_e"}, {"h"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashCString, {"sz", "s"}, {"n"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashSpeedTreeRippleGroup, {"s"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashFloat, {"f", "fl"}, {"a", "wd"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashInt64, {"a", "dw", "m_a", "m_cell", "n", "pt", "u"}, {"blend", "constraint", "follower", "pn", "sz"}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashRange, {"fl"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashBcQuat, {"q"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashPrsTransform, {"transform"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashMystery, {"id", "t"}, {}, useCommonOnly);
    addTypePrefixes(&typePrefixes, hashInterpolationInt32, {"path"}, {}, useCommonOnly);
    return typePrefixes;
}

void deduplicateStrings(std::vector<std::string>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::unordered_map<uint32_t, std::vector<uint32_t>> loadFilteredFieldTypes(const fs::path& path,
                                                                           const std::unordered_set<uint32_t>& relevantFields) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> mapping;
    std::ifstream input(path);
    if (!input) {
        return mapping;
    }
    std::string fieldHash;
    std::string typeHash;
    while (input >> fieldHash >> typeHash) {
        const uint32_t field = parseHexValue(fieldHash);
        if (relevantFields.find(field) == relevantFields.end()) {
            continue;
        }
        mapping[field].push_back(parseHexValue(typeHash));
    }
    return mapping;
}

void prepareFieldMode(SearchContext& context) {
    if (context.options.mode != Mode::Field) {
        return;
    }
    const bool explicitPositionZero = context.options.positionTokens.find(0) != context.options.positionTokens.end() &&
                                      !context.options.positionTokens[0].empty();

    if (context.options.noPrefix) {
        if (!context.options.literal && !explicitPositionZero) {
            std::vector<uint32_t> generated;
            generated.reserve(context.mainPool.size());
            for (uint32_t tokenId : context.mainPool) {
                generated.push_back(getOrCreateToken(context, lowercaseFirstCharacter(context.tokens[tokenId].text)));
            }
            context.options.positionTokens[0] = {};
            context.options.positionTokens[0].reserve(generated.size());
            for (uint32_t tokenId : generated) {
                context.options.positionTokens[0].push_back(context.tokens[tokenId].text);
            }
        }
        return;
    }

    if (explicitPositionZero && context.checksumProvided) {
        return;
    }

    context.typePrefixes = buildTypePrefixes(context.options.useCommonPrefixes);
    context.typePrefixSize = context.typePrefixes.size();
    context.usingFieldTypeMap = true;
    context.fieldToTypes = loadFilteredFieldTypes("field_types.txt", context.targetLookup);

    if (!explicitPositionZero) {
        std::unordered_set<std::string> prefixSet;
        for (const auto& [fieldHash, typeHashes] : context.fieldToTypes) {
            if (context.targetLookup.find(fieldHash) == context.targetLookup.end()) {
                continue;
            }
            for (uint32_t typeHash : typeHashes) {
                const auto prefixesIt = context.typePrefixes.find(typeHash);
                if (prefixesIt != context.typePrefixes.end() && !prefixesIt->second.empty()) {
                    prefixSet.insert(prefixesIt->second.begin(), prefixesIt->second.end());
                } else {
                    const auto fallbackIt = context.typePrefixes.find(0);
                    if (fallbackIt != context.typePrefixes.end()) {
                        prefixSet.insert(fallbackIt->second.begin(), fallbackIt->second.end());
                    }
                }
            }
        }
        context.options.positionTokens[0] = std::vector<std::string>(prefixSet.begin(), prefixSet.end());
        deduplicateStrings(context.options.positionTokens[0]);
        for (const std::string& token : context.options.positionTokens[0]) {
            getOrCreateToken(context, token);
        }
    }

    std::unordered_set<uint32_t> referencedTypes;
    for (uint32_t target : context.targets) {
        const auto it = context.fieldToTypes.find(target);
        if (it == context.fieldToTypes.end()) {
            continue;
        }
        referencedTypes.insert(it->second.begin(), it->second.end());
    }
    if (referencedTypes.size() == 1) {
        context.usingFieldTypeMap = false;
    }
}

LengthPlan buildLengthPlan(const SearchContext& context, uint32_t length) {
    LengthPlan plan;
    plan.length = length;
    plan.pools.reserve(length);
    plan.poolSizes.reserve(length);

    for (uint32_t position = 0; position < length; ++position) {
        std::vector<uint32_t> pool;
        if (position + 1 == length && context.options.suffixTokens.has_value()) {
            for (const std::string& token : *context.options.suffixTokens) {
                pool.push_back(context.tokenIds.at(token));
            }
        } else {
            const auto subdictIt = context.options.positionTokens.find(position);
            if (subdictIt != context.options.positionTokens.end()) {
                for (const std::string& token : subdictIt->second) {
                    pool.push_back(context.tokenIds.at(token));
                }
            } else {
                pool = context.mainPool;
            }
        }
        plan.poolSizes.push_back(pool.size());
        plan.pools.push_back(std::move(pool));
    }

    uint64_t suffixProduct = 1;
    size_t start = length;
    while (start > 0) {
        const uint64_t poolSize = plan.poolSizes[start - 1];
        if (poolSize == 0) {
            start -= 1;
            suffixProduct = 0;
            break;
        }
        if (suffixProduct > std::numeric_limits<uint64_t>::max() / poolSize) {
            break;
        }
        suffixProduct *= poolSize;
        start -= 1;
    }
    plan.gpuStartPosition = start;
    plan.suffixSearchSpace = suffixProduct;
    return plan;
}

std::vector<uint64_t> buildTargetTable(const std::vector<uint32_t>& targets, uint32_t* outMask) {
    size_t tableSize = 1;
    while (tableSize < std::max<size_t>(8, targets.size() * 4)) {
        tableSize <<= 1;
    }
    std::vector<uint64_t> table(tableSize, kEmptyTableEntry);
    const uint32_t mask = static_cast<uint32_t>(tableSize - 1);
    for (uint32_t indexInTargets = 0; indexInTargets < targets.size(); ++indexInTargets) {
        const uint32_t target = targets[indexInTargets];
        uint32_t index = (target * 2654435761u) & mask;
        while (table[index] != kEmptyTableEntry) {
            index = (index + 1u) & mask;
        }
        table[index] = (static_cast<uint64_t>(indexInTargets + 1u) << 32) | static_cast<uint64_t>(target);
    }
    *outMask = mask;
    return table;
}

struct CandidateBuild {
    std::string text;
    std::string firstToken;
};

CandidateBuild buildCandidateDetails(const SearchContext& context,
                                    const LengthPlan& plan,
                                    const std::vector<uint32_t>& prefixTokenIds,
                                    uint64_t suffixIndex) {
    std::vector<uint32_t> tokenIds = prefixTokenIds;
    tokenIds.reserve(plan.length);
    for (size_t position = plan.gpuStartPosition; position < plan.length; ++position) {
        const uint64_t poolSize = plan.poolSizes[position];
        const uint64_t digit = (poolSize == 0) ? 0 : (suffixIndex % poolSize);
        suffixIndex = (poolSize == 0) ? 0 : (suffixIndex / poolSize);
        tokenIds.push_back(plan.pools[position][digit]);
    }

    CandidateBuild candidate;
    if (!tokenIds.empty()) {
        candidate.firstToken = context.tokens[tokenIds.front()].text;
    }

    size_t totalLength = 0;
    for (uint32_t tokenId : tokenIds) {
        totalLength += context.tokens[tokenId].text.size();
    }
    candidate.text.reserve(totalLength);
    for (uint32_t tokenId : tokenIds) {
        candidate.text += context.tokens[tokenId].text;
    }
    return candidate;
}

bool containsHash(const std::unordered_set<uint32_t>& set, uint32_t value) {
    return set.find(value) != set.end();
}

bool passesPairedGate(const SearchContext& context, const std::string& candidate) {
    static const std::vector<std::string> prefixes = {"t", "pt", "ar"};
    static const std::vector<std::string> suffixes = {"", "s"};
    for (const std::string& prefix : prefixes) {
        for (const std::string& suffix : suffixes) {
            const uint32_t hash = hashString(prefix + candidate + suffix) & kFieldMask;
            if (containsHash(context.unfoundFieldLookup, hash)) {
                return true;
            }
        }
    }
    return false;
}

bool passesFieldTypeGate(const SearchContext& context, uint32_t fieldHash, std::string_view firstToken) {
    const auto prefixPool = context.options.positionTokens.find(0);
    if (!context.usingFieldTypeMap || prefixPool == context.options.positionTokens.end() || prefixPool->second.empty()) {
        return true;
    }

    uint32_t resolvedHash = fieldHash;
    auto it = context.fieldToTypes.find(resolvedHash);
    if (it == context.fieldToTypes.end() || it->second.empty()) {
        resolvedHash = 0;
        it = context.fieldToTypes.find(resolvedHash);
    }

    if (it == context.fieldToTypes.end()) {
        const auto fallbackIt = context.typePrefixes.find(0);
        return fallbackIt != context.typePrefixes.end() && fallbackIt->second.count(std::string(firstToken)) > 0;
    }

    for (uint32_t typeHash : it->second) {
        const auto prefixesIt = context.typePrefixes.find(typeHash);
        if (prefixesIt == context.typePrefixes.end() || prefixesIt->second.empty()) {
            return false;
        }
        return prefixesIt->second.count(std::string(firstToken)) > 0;
    }

    return false;
}

void prepareGpuFieldTypeGate(SearchContext& context) {
    context.prefixOrdinalByTokenId.clear();
    context.targetIndexByHash.clear();
    context.targetAllowedFirstTokenMasks.clear();
    context.canUseGpuFieldTypeGate = false;

    if (context.options.mode != Mode::Field || !context.usingFieldTypeMap) {
        return;
    }

    const auto prefixPoolIt = context.options.positionTokens.find(0);
    if (prefixPoolIt == context.options.positionTokens.end() || prefixPoolIt->second.empty() || prefixPoolIt->second.size() > 64) {
        return;
    }

    std::unordered_map<std::string, uint32_t> prefixOrdinalByText;
    for (uint32_t ordinal = 0; ordinal < prefixPoolIt->second.size(); ++ordinal) {
        const std::string& token = prefixPoolIt->second[ordinal];
        prefixOrdinalByText.emplace(token, ordinal);
        const auto tokenIt = context.tokenIds.find(token);
        if (tokenIt != context.tokenIds.end()) {
            context.prefixOrdinalByTokenId.emplace(tokenIt->second, ordinal);
        }
    }

    context.targetAllowedFirstTokenMasks.assign(context.targets.size(), 0ull);
    for (uint32_t targetIndex = 0; targetIndex < context.targets.size(); ++targetIndex) {
        const uint32_t targetHash = context.targets[targetIndex];
        context.targetIndexByHash.emplace(targetHash, targetIndex);

        uint32_t resolvedHash = targetHash;
        auto fieldIt = context.fieldToTypes.find(resolvedHash);
        if (fieldIt == context.fieldToTypes.end() || fieldIt->second.empty()) {
            resolvedHash = 0;
            fieldIt = context.fieldToTypes.find(resolvedHash);
        }

        const std::unordered_set<std::string>* allowedPrefixes = nullptr;
        if (fieldIt == context.fieldToTypes.end()) {
            const auto fallbackIt = context.typePrefixes.find(0);
            if (fallbackIt != context.typePrefixes.end()) {
                allowedPrefixes = &fallbackIt->second;
            }
        } else {
            const uint32_t typeHash = fieldIt->second.front();
            const auto prefixesIt = context.typePrefixes.find(typeHash);
            if (prefixesIt != context.typePrefixes.end() && !prefixesIt->second.empty()) {
                allowedPrefixes = &prefixesIt->second;
            }
        }

        if (allowedPrefixes == nullptr) {
            continue;
        }

        uint64_t mask = 0;
        for (const std::string& prefix : *allowedPrefixes) {
            const auto ordinalIt = prefixOrdinalByText.find(prefix);
            if (ordinalIt != prefixOrdinalByText.end()) {
                mask |= (1ull << ordinalIt->second);
            }
        }
        context.targetAllowedFirstTokenMasks[targetIndex] = mask;
    }

    context.canUseGpuFieldTypeGate = true;
}

bool passesAllGates(const SearchContext& context,
                    uint32_t matchedHash,
                    const std::string& candidate,
                    std::string_view firstToken) {
    if (context.options.mode == Mode::Type && context.options.paired && !passesPairedGate(context, candidate)) {
        return false;
    }
    if (context.options.mode == Mode::Field && !passesFieldTypeGate(context, matchedHash, firstToken)) {
        return false;
    }
    return true;
}

std::vector<uint32_t> readTargetsFromStdin() {
    std::vector<uint32_t> targets;
#if defined(_WIN32)
    if (_isatty(_fileno(stdin)) != 0) {
        return targets;
    }
#elif defined(__unix__) || defined(__APPLE__)
    if (isatty(fileno(stdin)) != 0) {
        return targets;
    }
#endif
    std::string token;
    while (std::cin >> token) {
        targets.push_back(parseHexValue(token));
    }
    return targets;
}

void sortAndUnique(std::vector<uint32_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::string formatHashRate(double candidatesPerSecond) {
    static constexpr std::array<std::pair<double, const char*>, 5> units = {{
        {1'000'000'000'000.0, "T/s"},
        {1'000'000'000.0, "G/s"},
        {1'000'000.0, "M/s"},
        {1'000.0, "K/s"},
        {1.0, "/s"},
    }};

    std::ostringstream stream;
    stream << std::fixed;
    for (const auto& [threshold, suffix] : units) {
        if (candidatesPerSecond < threshold) {
            continue;
        }
        const double scaled = candidatesPerSecond / threshold;
        const int precision = (scaled >= 100.0 || threshold == 1.0) ? 0 : (scaled >= 10.0 ? 1 : 2);
        stream << std::setprecision(precision) << scaled << suffix;
        return stream.str();
    }
    stream << std::setprecision(0) << 0.0 << "/s";
    return stream.str();
}

void printUsageError(const std::string& message) {
    std::cerr << message << '\n';
}

Options parseOptions(int argc, char** argv) {
    Options options;
    options.threads = std::clamp<uint32_t>(std::max(1u, std::thread::hardware_concurrency()), 1u, 64u);
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--field") {
            options.mode = Mode::Field;
        } else if (argument == "--gbid") {
            options.mode = Mode::Gbid;
        } else if (argument == "--dict") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Error: --dict requires an argument");
            }
            options.dictArg = argv[++index];
        } else if (argument == "--english") {
            options.useEnglish = true;
        } else if (argument == "--expanded") {
            options.useExpanded = true;
        } else if (argument == "--no-dict") {
            options.noDict = true;
        } else if (argument == "--literal") {
            options.literal = true;
        } else if (argument == "--words-only") {
            options.wordsOnly = true;
        } else if (argument == "--allow-all-caps") {
            options.allowAllCaps = true;
        } else if (argument == "--prefix") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Error: --prefix requires an argument");
            }
            options.positionTokens[0] = parseTokenArgument(argv[++index]);
        } else if (argument == "--suffix") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Error: --suffix requires an argument");
            }
            options.suffixTokens = parseTokenArgument(argv[++index]);
        } else if (argument == "--subdict") {
            if (index + 2 >= argc) {
                throw std::runtime_error("Error: --subdict requires two arguments");
            }
            const size_t position = static_cast<size_t>(std::stoul(argv[++index]));
            options.positionTokens[position] = parseTokenArgument(argv[++index]);
        } else if (argument == "--no-prefix") {
            options.positionTokens.erase(0);
            options.noPrefix = true;
        } else if (argument == "--min") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Error: --min requires an argument");
            }
            const uint32_t parsed = static_cast<uint32_t>(std::stoul(argv[++index]));
            if (parsed >= 1 && parsed < 64) {
                options.minLength = parsed;
                options.minSpecified = true;
            }
        } else if (argument == "--max") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Error: --max requires an argument");
            }
            const uint32_t parsed = static_cast<uint32_t>(std::stoul(argv[++index]));
            if (parsed >= 1 && parsed < 64) {
                options.maxLength = parsed;
                options.maxSpecified = true;
            }
        } else if (argument == "--threads") {
            if (index + 1 >= argc) {
                throw std::runtime_error("Error: --threads requires an argument");
            }
            options.threads = std::clamp<uint32_t>(static_cast<uint32_t>(std::stoul(argv[++index])), 1u, 64u);
        } else if (argument == "--log") {
            options.logMatches = true;
        } else if (argument == "--force") {
            options.force = true;
        } else if (argument == "--paired") {
            options.paired = true;
        } else if (argument == "--common") {
            options.useCommonPrefixes = true;
        } else if (argument == "--uncommon") {
            options.useCommonPrefixes = false;
        } else if (!argument.empty() && argument[0] == '-') {
            throw std::runtime_error("Error: Unknown option: " + argument);
        } else {
            options.explicitTargets.push_back(parseHexValue(argument));
        }
    }
    return options;
}

void buildPow33(SearchContext& context) {
    size_t maxLength = 1;
    for (const Token& token : context.tokens) {
        maxLength = std::max<size_t>(maxLength, token.length);
    }
    context.pow33.assign(maxLength + 1, 1u);
    for (size_t index = 1; index < context.pow33.size(); ++index) {
        context.pow33[index] = context.pow33[index - 1] * 33u;
    }
}

void prepareTargets(SearchContext& context) {
    context.unfoundFieldLookup.clear();
    for (uint32_t hash : loadHexList("unfound_field_hashes.txt")) {
        context.unfoundFieldLookup.insert(hash);
    }

    std::vector<uint32_t> repoUnfound;
    if (context.options.mode == Mode::Type) {
        repoUnfound = loadHexList("unfound_hashes.txt");
    } else if (context.options.mode == Mode::Field) {
        repoUnfound = loadHexList("unfound_field_hashes.txt");
    }
    context.relevantUnfoundLookup.clear();
    for (uint32_t hash : repoUnfound) {
        context.relevantUnfoundLookup.insert(hash);
    }

    std::vector<uint32_t> combinedTargets = context.options.explicitTargets;
    std::vector<uint32_t> stdinTargets = readTargetsFromStdin();
    combinedTargets.insert(combinedTargets.end(), stdinTargets.begin(), stdinTargets.end());
    sortAndUnique(combinedTargets);
    context.checksumProvided = !combinedTargets.empty();

    if (combinedTargets.empty()) {
        context.targets = repoUnfound;
    } else {
        if (!context.options.force) {
            std::vector<uint32_t> filtered;
            filtered.reserve(combinedTargets.size());
            for (uint32_t value : combinedTargets) {
                if (containsHash(context.relevantUnfoundLookup, value)) {
                    filtered.push_back(value);
                } else {
                    std::cout << "removing already known hash: " << OutputSink::toHex(value) << '\n';
                }
            }
            std::cout << std::dec << '\n';
            combinedTargets.swap(filtered);
        }
        context.targets = combinedTargets;
    }

    if (context.options.mode != Mode::Type) {
        context.options.paired = false;
    }

    sortAndUnique(context.targets);
    context.targetLookup.clear();
    for (uint32_t target : context.targets) {
        context.targetLookup.insert(target);
    }
}

bool ensureDeviceCapacity(void** pointer, size_t* capacityBytes, size_t requiredBytes) {
    if (requiredBytes <= *capacityBytes) {
        return true;
    }
    cudaFree(*pointer);
    *pointer = nullptr;
    *capacityBytes = 0;
    if (requiredBytes == 0) {
        return true;
    }
    if (cudaMalloc(pointer, requiredBytes) != cudaSuccess) {
        return false;
    }
    *capacityBytes = requiredBytes;
    return true;
}

bool initializeGpuBuffers(const SearchContext& context,
                          GpuBuffers* buffers) {
    uint32_t targetMask = 0;
    std::vector<uint64_t> targetTable = buildTargetTable(context.targets, &targetMask);

    buffers->targetMask = targetMask;
    buffers->fieldGateEnabled = context.canUseGpuFieldTypeGate;

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, device);
    buffers->blockSize = 256;
    buffers->gridSize = std::max(1, props.multiProcessorCount * static_cast<int>(context.options.threads) * 2);

    const auto allocate = [](void** pointer, size_t bytes) {
        return cudaMalloc(pointer, bytes) == cudaSuccess;
    };
    const auto allocatePinned = [](void** pointer, size_t bytes) {
        return cudaMallocHost(pointer, bytes) == cudaSuccess;
    };

    if (!allocate(reinterpret_cast<void**>(&buffers->d_pow33), context.pow33.size() * sizeof(uint32_t)) ||
        !allocate(reinterpret_cast<void**>(&buffers->d_targetTable), targetTable.size() * sizeof(uint64_t)) ||
        (buffers->fieldGateEnabled && !allocate(reinterpret_cast<void**>(&buffers->d_targetPrefixMasks), context.targetAllowedFirstTokenMasks.size() * sizeof(uint64_t)))) {
        return false;
    }

    for (auto& slot : buffers->slots) {
        if (cudaStreamCreate(&slot.stream) != cudaSuccess ||
            !allocate(reinterpret_cast<void**>(&slot.d_matches), kMatchBufferCapacity * sizeof(DeviceMatch)) ||
            !allocate(reinterpret_cast<void**>(&slot.d_matchCount), sizeof(uint32_t)) ||
            !allocate(reinterpret_cast<void**>(&slot.d_overflow), sizeof(uint32_t)) ||
            !allocatePinned(reinterpret_cast<void**>(&slot.hostMatches), kMatchBufferCapacity * sizeof(DeviceMatch)) ||
            !allocatePinned(reinterpret_cast<void**>(&slot.hostMatchCount), sizeof(uint32_t)) ||
            !allocatePinned(reinterpret_cast<void**>(&slot.hostOverflow), sizeof(uint32_t))) {
            return false;
        }
    }

    cudaMemcpy(buffers->d_pow33, context.pow33.data(), context.pow33.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers->d_targetTable, targetTable.data(), targetTable.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (buffers->fieldGateEnabled && !context.targetAllowedFirstTokenMasks.empty()) {
        cudaMemcpy(buffers->d_targetPrefixMasks,
                   context.targetAllowedFirstTokenMasks.data(),
                   context.targetAllowedFirstTokenMasks.size() * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool uploadLengthPlanToGpu(const LengthPlan& plan,
                           const SearchContext& context,
                           GpuBuffers* buffers) {
    std::vector<uint32_t> poolOffsets;
    std::vector<uint32_t> poolSizes;
    std::vector<DeviceToken> poolTokens;
    for (size_t position = plan.gpuStartPosition; position < plan.length; ++position) {
        poolOffsets.push_back(static_cast<uint32_t>(poolTokens.size()));
        poolSizes.push_back(static_cast<uint32_t>(plan.pools[position].size()));
        for (uint32_t tokenId : plan.pools[position]) {
            const Token& token = context.tokens[tokenId];
            DeviceToken deviceToken{};
            deviceToken.length = token.length;
            deviceToken.hash = token.hash;
            poolTokens.push_back(deviceToken);
        }
    }

    if (!ensureDeviceCapacity(reinterpret_cast<void**>(&buffers->d_poolOffsets),
                              &buffers->poolOffsetsCapacity,
                              poolOffsets.size() * sizeof(uint32_t)) ||
        !ensureDeviceCapacity(reinterpret_cast<void**>(&buffers->d_poolSizes),
                              &buffers->poolSizesCapacity,
                              poolSizes.size() * sizeof(uint32_t)) ||
        !ensureDeviceCapacity(reinterpret_cast<void**>(&buffers->d_poolTokens),
                              &buffers->poolTokensCapacity,
                              poolTokens.size() * sizeof(DeviceToken))) {
        return false;
    }

    if (!poolOffsets.empty()) {
        cudaMemcpy(buffers->d_poolOffsets, poolOffsets.data(), poolOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(buffers->d_poolSizes, poolSizes.data(), poolSizes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    if (!poolTokens.empty()) {
        cudaMemcpy(buffers->d_poolTokens, poolTokens.data(), poolTokens.size() * sizeof(DeviceToken), cudaMemcpyHostToDevice);
    }
    return cudaGetLastError() == cudaSuccess;
}

void releaseGpuBuffers(GpuBuffers* buffers) {
    cudaFree(buffers->d_pow33);
    cudaFree(buffers->d_targetTable);
    cudaFree(buffers->d_targetPrefixMasks);
    cudaFree(buffers->d_poolOffsets);
    cudaFree(buffers->d_poolSizes);
    cudaFree(buffers->d_poolTokens);
    for (auto& slot : buffers->slots) {
        cudaFree(slot.d_matches);
        cudaFree(slot.d_matchCount);
        cudaFree(slot.d_overflow);
        if (slot.hostMatches != nullptr) {
            cudaFreeHost(slot.hostMatches);
        }
        if (slot.hostMatchCount != nullptr) {
            cudaFreeHost(slot.hostMatchCount);
        }
        if (slot.hostOverflow != nullptr) {
            cudaFreeHost(slot.hostOverflow);
        }
        if (slot.stream != nullptr) {
            cudaStreamDestroy(slot.stream);
        }
    }
}

__device__ __forceinline__ uint32_t deviceFindTargetIndex(const uint64_t* table, uint32_t mask, uint32_t value) {
    uint32_t index = (value * 2654435761u) & mask;
    while (true) {
        const uint64_t entry = table[index];
        if (entry == kEmptyTableEntry) {
            return UINT32_MAX;
        }
        if (static_cast<uint32_t>(entry) == value) {
            return static_cast<uint32_t>((entry >> 32) - 1u);
        }
        index = (index + 1u) & mask;
    }
}

template <int SuffixLength>
__global__ void searchKernelFixed(uint32_t mode,
                                  uint64_t startIndex,
                                  uint64_t count,
                                  uint32_t prefixHash,
                                  const uint32_t* pow33,
                                  const uint64_t* targetTable,
                                  const uint64_t* targetPrefixMasks,
                                  uint32_t targetMask,
                                  int32_t prefixFirstTokenOrdinal,
                                  const uint32_t* poolOffsets,
                                  const uint32_t* poolSizes,
                                  const DeviceToken* poolTokens,
                                  DeviceMatch* matches,
                                  uint32_t matchCapacity,
                                  uint32_t* matchCount,
                                  uint32_t* overflowFlag) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t endIndex = startIndex + count;
    for (uint64_t suffixIndex = startIndex + blockIdx.x * blockDim.x + threadIdx.x;
         suffixIndex < endIndex;
         suffixIndex += stride) {
        uint64_t index = suffixIndex;
        uint32_t hash = prefixHash;
        int32_t firstTokenOrdinal = prefixFirstTokenOrdinal;

        if constexpr (SuffixLength >= 1) {
            const uint32_t poolSize0 = poolSizes[0];
            const uint32_t digit0 = static_cast<uint32_t>(index % poolSize0);
            index /= poolSize0;
            if (firstTokenOrdinal < 0) {
                firstTokenOrdinal = static_cast<int32_t>(digit0);
            }
            const DeviceToken token0 = poolTokens[poolOffsets[0] + digit0];
            hash = combineHash(hash, token0.hash, pow33[token0.length]);
        }
        if constexpr (SuffixLength >= 2) {
            const uint32_t poolSize1 = poolSizes[1];
            const uint32_t digit1 = static_cast<uint32_t>(index % poolSize1);
            index /= poolSize1;
            const DeviceToken token1 = poolTokens[poolOffsets[1] + digit1];
            hash = combineHash(hash, token1.hash, pow33[token1.length]);
        }
        if constexpr (SuffixLength >= 3) {
            const uint32_t poolSize2 = poolSizes[2];
            const uint32_t digit2 = static_cast<uint32_t>(index % poolSize2);
            const DeviceToken token2 = poolTokens[poolOffsets[2] + digit2];
            hash = combineHash(hash, token2.hash, pow33[token2.length]);
        }

        const uint32_t visibleHash = (mode == 1u) ? (hash & kFieldMask) : hash;
        const uint32_t targetIndex = deviceFindTargetIndex(targetTable, targetMask, visibleHash);
        if (targetIndex == UINT32_MAX) {
            continue;
        }
        if (targetPrefixMasks != nullptr) {
            if (firstTokenOrdinal < 0 || (targetPrefixMasks[targetIndex] & (1ull << firstTokenOrdinal)) == 0) {
                continue;
            }
        }
        const uint32_t slot = atomicAdd(matchCount, 1u);
        if (slot < matchCapacity) {
            matches[slot].hash = visibleHash;
            matches[slot].suffixIndex = suffixIndex;
        } else {
            *overflowFlag = 1u;
        }
    }
}

__global__ void searchKernel(uint32_t mode,
                             uint32_t suffixLength,
                             uint64_t startIndex,
                             uint64_t count,
                             uint32_t prefixHash,
                             const uint32_t* pow33,
                             const uint64_t* targetTable,
                             const uint64_t* targetPrefixMasks,
                             uint32_t targetMask,
                             int32_t prefixFirstTokenOrdinal,
                             const uint32_t* poolOffsets,
                             const uint32_t* poolSizes,
                             const DeviceToken* poolTokens,
                             DeviceMatch* matches,
                             uint32_t matchCapacity,
                             uint32_t* matchCount,
                             uint32_t* overflowFlag) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t endIndex = startIndex + count;
    for (uint64_t suffixIndex = startIndex + blockIdx.x * blockDim.x + threadIdx.x;
         suffixIndex < endIndex;
         suffixIndex += stride) {
        uint64_t index = suffixIndex;
        uint32_t hash = prefixHash;
        int32_t firstTokenOrdinal = prefixFirstTokenOrdinal;
        for (uint32_t position = 0; position < suffixLength; ++position) {
            const uint32_t poolSize = poolSizes[position];
            const uint32_t digit = static_cast<uint32_t>(index % poolSize);
            index /= poolSize;
            if (firstTokenOrdinal < 0) {
                firstTokenOrdinal = static_cast<int32_t>(digit);
            }
            const DeviceToken token = poolTokens[poolOffsets[position] + digit];
            hash = combineHash(hash, token.hash, pow33[token.length]);
        }
        const uint32_t visibleHash = (mode == 1u) ? (hash & kFieldMask) : hash;
        const uint32_t targetIndex = deviceFindTargetIndex(targetTable, targetMask, visibleHash);
        if (targetIndex == UINT32_MAX) {
            continue;
        }
        if (targetPrefixMasks != nullptr) {
            if (firstTokenOrdinal < 0 || (targetPrefixMasks[targetIndex] & (1ull << firstTokenOrdinal)) == 0) {
                continue;
            }
        }
        const uint32_t slot = atomicAdd(matchCount, 1u);
        if (slot < matchCapacity) {
            matches[slot].hash = visibleHash;
            matches[slot].suffixIndex = suffixIndex;
        } else {
            *overflowFlag = 1u;
        }
    }
}

int32_t resolvePrefixFirstTokenOrdinal(const SearchContext& context,
                                       const std::vector<uint32_t>& prefixTokenIds) {
    if (prefixTokenIds.empty()) {
        return -1;
    }
    const auto ordinalIt = context.prefixOrdinalByTokenId.find(prefixTokenIds.front());
    if (ordinalIt == context.prefixOrdinalByTokenId.end()) {
        return -1;
    }
    return static_cast<int32_t>(ordinalIt->second);
}

bool launchGpuSuffixRange(const SearchContext& context,
                          const LengthPlan& plan,
                          const GpuBuffers* buffers,
                          GpuBuffers::BatchSlot* slot,
                          uint32_t prefixHash,
                          uint64_t startIndex,
                          uint64_t count,
                          int32_t prefixFirstTokenOrdinal) {
    if (buffers == nullptr || slot == nullptr) {
        return false;
    }
    uint32_t zero = 0;
    if (cudaMemcpyAsync(slot->d_matchCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, slot->stream) != cudaSuccess ||
        cudaMemcpyAsync(slot->d_overflow, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, slot->stream) != cudaSuccess) {
        return false;
    }

    const uint32_t suffixLength = static_cast<uint32_t>(plan.length - plan.gpuStartPosition);
    const uint64_t* targetPrefixMasks = buffers->fieldGateEnabled ? buffers->d_targetPrefixMasks : nullptr;
    if (suffixLength == 1) {
        searchKernelFixed<1><<<buffers->gridSize, buffers->blockSize, 0, slot->stream>>>(
            context.options.mode == Mode::Field ? 1u : 0u,
            startIndex,
            count,
            prefixHash,
            buffers->d_pow33,
            buffers->d_targetTable,
            targetPrefixMasks,
            buffers->targetMask,
            prefixFirstTokenOrdinal,
            buffers->d_poolOffsets,
            buffers->d_poolSizes,
            buffers->d_poolTokens,
            slot->d_matches,
            kMatchBufferCapacity,
            slot->d_matchCount,
            slot->d_overflow);
    } else if (suffixLength == 2) {
        searchKernelFixed<2><<<buffers->gridSize, buffers->blockSize, 0, slot->stream>>>(
            context.options.mode == Mode::Field ? 1u : 0u,
            startIndex,
            count,
            prefixHash,
            buffers->d_pow33,
            buffers->d_targetTable,
            targetPrefixMasks,
            buffers->targetMask,
            prefixFirstTokenOrdinal,
            buffers->d_poolOffsets,
            buffers->d_poolSizes,
            buffers->d_poolTokens,
            slot->d_matches,
            kMatchBufferCapacity,
            slot->d_matchCount,
            slot->d_overflow);
    } else if (suffixLength == 3) {
        searchKernelFixed<3><<<buffers->gridSize, buffers->blockSize, 0, slot->stream>>>(
            context.options.mode == Mode::Field ? 1u : 0u,
            startIndex,
            count,
            prefixHash,
            buffers->d_pow33,
            buffers->d_targetTable,
            targetPrefixMasks,
            buffers->targetMask,
            prefixFirstTokenOrdinal,
            buffers->d_poolOffsets,
            buffers->d_poolSizes,
            buffers->d_poolTokens,
            slot->d_matches,
            kMatchBufferCapacity,
            slot->d_matchCount,
            slot->d_overflow);
    } else {
        searchKernel<<<buffers->gridSize, buffers->blockSize, 0, slot->stream>>>(
            context.options.mode == Mode::Field ? 1u : 0u,
            suffixLength,
            startIndex,
            count,
            prefixHash,
            buffers->d_pow33,
            buffers->d_targetTable,
            targetPrefixMasks,
            buffers->targetMask,
            prefixFirstTokenOrdinal,
            buffers->d_poolOffsets,
            buffers->d_poolSizes,
            buffers->d_poolTokens,
            slot->d_matches,
            kMatchBufferCapacity,
            slot->d_matchCount,
            slot->d_overflow);
    }
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    return cudaMemcpyAsync(slot->hostMatchCount,
                           slot->d_matchCount,
                           sizeof(uint32_t),
                           cudaMemcpyDeviceToHost,
                           slot->stream) == cudaSuccess &&
           cudaMemcpyAsync(slot->hostOverflow,
                           slot->d_overflow,
                           sizeof(uint32_t),
                           cudaMemcpyDeviceToHost,
                           slot->stream) == cudaSuccess;
}

bool emitGpuMatches(const SearchContext& context,
                    const LengthPlan& plan,
                    const GpuBuffers* buffers,
                    GpuBuffers::BatchSlot* slot,
                    const std::vector<uint32_t>& prefixTokenIds,
                    uint32_t matchCount,
                    OutputSink* output) {
    if (matchCount == 0) {
        return true;
    }
    if (cudaMemcpyAsync(slot->hostMatches,
                        slot->d_matches,
                        matchCount * sizeof(DeviceMatch),
                        cudaMemcpyDeviceToHost,
                        slot->stream) != cudaSuccess ||
        cudaStreamSynchronize(slot->stream) != cudaSuccess) {
        return false;
    }
    const bool fieldGateAlreadyApplied = buffers->fieldGateEnabled && context.options.mode == Mode::Field;
    for (uint32_t index = 0; index < matchCount; ++index) {
        const DeviceMatch& match = slot->hostMatches[index];
        const CandidateBuild candidate = buildCandidateDetails(context, plan, prefixTokenIds, match.suffixIndex);
        if (context.options.mode == Mode::Field) {
            if (!fieldGateAlreadyApplied && !passesAllGates(context, match.hash, candidate.text, candidate.firstToken)) {
                continue;
            }
        } else if (!passesAllGates(context, match.hash, candidate.text, candidate.firstToken)) {
            continue;
        }
        output->emit(match.hash, candidate.text);
    }
    return true;
}

bool processGpuSuffixRangeSync(const SearchContext& context,
                               const LengthPlan& plan,
                               const GpuBuffers* buffers,
                               GpuBuffers::BatchSlot* slot,
                               const std::vector<uint32_t>& prefixTokenIds,
                               uint32_t prefixHash,
                               uint64_t startIndex,
                               uint64_t count,
                               int32_t prefixFirstTokenOrdinal,
                               OutputSink* output);

bool finalizeGpuSuffixRange(const SearchContext& context,
                            const LengthPlan& plan,
                            const GpuBuffers* buffers,
                            GpuBuffers::BatchSlot* slot,
                            const std::vector<uint32_t>& prefixTokenIds,
                            uint32_t prefixHash,
                            uint64_t startIndex,
                            uint64_t count,
                            int32_t prefixFirstTokenOrdinal,
                            OutputSink* output) {
    if (slot == nullptr || cudaStreamSynchronize(slot->stream) != cudaSuccess) {
        return false;
    }
    const uint32_t matchCount = *slot->hostMatchCount;
    const uint32_t overflow = *slot->hostOverflow;
    if (overflow != 0 && count > 1) {
        const uint64_t half = count / 2;
        return processGpuSuffixRangeSync(context,
                                         plan,
                                         buffers,
                                         slot,
                                         prefixTokenIds,
                                         prefixHash,
                                         startIndex,
                                         half,
                                         prefixFirstTokenOrdinal,
                                         output) &&
               processGpuSuffixRangeSync(context,
                                         plan,
                                         buffers,
                                         slot,
                                         prefixTokenIds,
                                         prefixHash,
                                         startIndex + half,
                                         count - half,
                                         prefixFirstTokenOrdinal,
                                         output);
    }
    return emitGpuMatches(context, plan, buffers, slot, prefixTokenIds, matchCount, output);
}

bool processGpuSuffixRangeSync(const SearchContext& context,
                               const LengthPlan& plan,
                               const GpuBuffers* buffers,
                               GpuBuffers::BatchSlot* slot,
                               const std::vector<uint32_t>& prefixTokenIds,
                               uint32_t prefixHash,
                               uint64_t startIndex,
                               uint64_t count,
                               int32_t prefixFirstTokenOrdinal,
                               OutputSink* output) {
    return launchGpuSuffixRange(context,
                                plan,
                                buffers,
                                slot,
                                prefixHash,
                                startIndex,
                                count,
                                prefixFirstTokenOrdinal) &&
           finalizeGpuSuffixRange(context,
                                  plan,
                                  buffers,
                                  slot,
                                  prefixTokenIds,
                                  prefixHash,
                                  startIndex,
                                  count,
                                  prefixFirstTokenOrdinal,
                                  output);
}

bool runGpuSearch(const SearchContext& context,
                  const LengthPlan& plan,
                  GpuBuffers* buffers,
                  const std::vector<uint32_t>& prefixTokenIds,
                  uint32_t prefixHash,
                  OutputSink* output) {
    if (buffers == nullptr) {
        return false;
    }
    const int32_t prefixFirstTokenOrdinal = resolvePrefixFirstTokenOrdinal(context, prefixTokenIds);
    struct PendingBatch {
        GpuBuffers::BatchSlot* slot = nullptr;
        uint64_t startIndex = 0;
        uint64_t count = 0;
        bool valid = false;
    } pending;

    bool ok = true;
    size_t nextSlotIndex = 0;
    for (uint64_t start = 0; start < plan.suffixSearchSpace && !g_interrupted.load(std::memory_order_relaxed); start += kDefaultBatchSize) {
        const uint64_t count = std::min<uint64_t>(kDefaultBatchSize, plan.suffixSearchSpace - start);
        GpuBuffers::BatchSlot* slot = &buffers->slots[nextSlotIndex];
        ok = launchGpuSuffixRange(context,
                                  plan,
                                  buffers,
                                  slot,
                                  prefixHash,
                                  start,
                                  count,
                                  prefixFirstTokenOrdinal);
        if (!ok) {
            break;
        }
        if (pending.valid) {
            ok = finalizeGpuSuffixRange(context,
                                        plan,
                                        buffers,
                                        pending.slot,
                                        prefixTokenIds,
                                        prefixHash,
                                        pending.startIndex,
                                        pending.count,
                                        prefixFirstTokenOrdinal,
                                        output);
        }
        if (!ok) {
            break;
        }
        pending.slot = slot;
        pending.startIndex = start;
        pending.count = count;
        pending.valid = true;
        nextSlotIndex = (nextSlotIndex + 1) % buffers->slots.size();
    }
    if (ok && pending.valid) {
        ok = finalizeGpuSuffixRange(context,
                                    plan,
                                    buffers,
                                    pending.slot,
                                    prefixTokenIds,
                                    prefixHash,
                                    pending.startIndex,
                                    pending.count,
                                    prefixFirstTokenOrdinal,
                                    output);
    }
    return ok;
}

void runCpuSuffixRange(const SearchContext& context,
                       const LengthPlan& plan,
                       const std::vector<uint32_t>& prefixTokenIds,
                       uint32_t prefixHash,
                       uint64_t startIndex,
                       uint64_t count,
                       OutputSink* output) {
    const uint32_t workerCount = std::max<uint32_t>(1u, context.options.threads);
    std::vector<std::thread> workers;
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&, worker]() {
            for (uint64_t offset = worker; offset < count && !g_interrupted.load(std::memory_order_relaxed); offset += workerCount) {
                uint64_t index = startIndex + offset;
                uint64_t decode = index;
                uint32_t hash = prefixHash;
                for (size_t position = plan.gpuStartPosition; position < plan.length; ++position) {
                    const uint64_t poolSize = plan.poolSizes[position];
                    const uint64_t digit = decode % poolSize;
                    decode /= poolSize;
                    const Token& token = context.tokens[plan.pools[position][digit]];
                    hash = combineHash(hash, token.hash, context.pow33[token.length]);
                }
                const uint32_t visibleHash = (context.options.mode == Mode::Field) ? (hash & kFieldMask) : hash;
                if (containsHash(context.targetLookup, visibleHash)) {
                    const CandidateBuild candidate = buildCandidateDetails(context, plan, prefixTokenIds, index);
                    if (!passesAllGates(context, visibleHash, candidate.text, candidate.firstToken)) {
                        continue;
                    }
                    output->emit(visibleHash, candidate.text);
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

bool hasUsableGpu() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return deviceCount > 0;
}

SearchCount searchPrefixes(const SearchContext& context,
                          const LengthPlan& plan,
                          size_t position,
                          std::vector<uint32_t>& prefixTokenIds,
                          uint32_t prefixHash,
                          GpuBuffers* gpuBuffers,
                          bool useGpu,
                          OutputSink* output) {
    if (g_interrupted.load(std::memory_order_relaxed)) {
        return {};
    }
    if (position == plan.gpuStartPosition) {
        if (plan.suffixSearchSpace == 0) {
            return {};
        }
        if (useGpu) {
            if (!runGpuSearch(context, plan, gpuBuffers, prefixTokenIds, prefixHash, output)) {
                runCpuSuffixRange(context, plan, prefixTokenIds, prefixHash, 0, plan.suffixSearchSpace, output);
            }
        } else {
            runCpuSuffixRange(context, plan, prefixTokenIds, prefixHash, 0, plan.suffixSearchSpace, output);
        }
        SearchCount searched;
        searched += plan.suffixSearchSpace;
        return searched;
    }

    SearchCount searched;
    for (uint32_t tokenId : plan.pools[position]) {
        prefixTokenIds.push_back(tokenId);
        const Token& token = context.tokens[tokenId];
        const uint32_t nextHash = combineHash(prefixHash, token.hash, context.pow33[token.length]);
        searched += searchPrefixes(context, plan, position + 1, prefixTokenIds, nextHash, gpuBuffers, useGpu, output);
        prefixTokenIds.pop_back();
        if (g_interrupted.load(std::memory_order_relaxed)) {
            break;
        }
    }
    return searched;
}

double toDouble(const SearchCount& value) {
    return static_cast<double>(value.high) * 18446744073709551616.0 + static_cast<double>(value.low);
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    SearchContext context;
    try {
        context.options = parseOptions(argc, argv);
    } catch (const std::exception& error) {
        printUsageError(error.what());
        return 1;
    }

    prepareTargets(context);

    const DictionaryBuild build = buildMainPool(context);
    context.mainPool = build.mainPool;

    for (auto& entry : context.options.positionTokens) {
        for (const std::string& token : entry.second) {
            getOrCreateToken(context, token);
        }
    }

    if (context.options.suffixTokens.has_value()) {
        for (const std::string& token : *context.options.suffixTokens) {
            getOrCreateToken(context, token);
        }
    }

    prepareFieldMode(context);
    prepareGpuFieldTypeGate(context);
    buildPow33(context);

    if (context.targets.empty()) {
        return 0;
    }

    const bool useGpu = hasUsableGpu();
    GpuBuffers gpuBuffers;
    GpuBuffers* gpuBuffersPtr = nullptr;
    if (useGpu) {
        if (!initializeGpuBuffers(context, &gpuBuffers)) {
            releaseGpuBuffers(&gpuBuffers);
            return 1;
        }
        gpuBuffersPtr = &gpuBuffers;
    }
    OutputSink output(context.options.logMatches, context.options.mode);

    const size_t prefixSize = [&]() {
        const auto it = context.options.positionTokens.find(0);
        return (it == context.options.positionTokens.end()) ? 0u : static_cast<unsigned>(it->second.size());
    }();
    const size_t suffixSize = context.options.suffixTokens.has_value() ? context.options.suffixTokens->size() : 0u;

    if (context.options.minSpecified) {
        std::cerr << "Using min of " << (context.options.minLength - 1) << '\n';
    }
    if (context.options.maxSpecified) {
        std::cerr << "Using max of " << context.options.maxLength << '\n';
    }
    if (context.options.minSpecified || context.options.maxSpecified) {
        std::cerr << '\n';
    }
    std::cerr << "Type prefix size: " << context.typePrefixSize << '\n';
    std::cerr << "Prefix size: " << prefixSize << '\n';
    std::cerr << "Dictionary size: " << context.mainPool.size() << '\n';
    std::cerr << "Suffix size: " << suffixSize << '\n';
    std::cerr << "Matching " << context.targets.size() << " hashes." << '\n';
    if (context.options.threads > 1) {
        std::cerr << "Using " << context.options.threads << " workers." << '\n';
    }

    SearchCount searched;
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t length = context.options.minLength; length <= context.options.maxLength; ++length) {
        if (g_interrupted.load(std::memory_order_relaxed)) {
            break;
        }
        std::cerr << "Length: " << length << '\n';
        const LengthPlan plan = buildLengthPlan(context, length);
        if (plan.suffixSearchSpace == 0) {
            continue;
        }
        if (gpuBuffersPtr != nullptr && !uploadLengthPlanToGpu(plan, context, gpuBuffersPtr)) {
            releaseGpuBuffers(gpuBuffersPtr);
            return 1;
        }
        
        std::vector<uint32_t> prefixTokenIds;
        prefixTokenIds.reserve(plan.gpuStartPosition);
        searched += searchPrefixes(context, plan, 0, prefixTokenIds, 0u, gpuBuffersPtr, useGpu, &output);
    }

    if (gpuBuffersPtr != nullptr) {
        releaseGpuBuffers(gpuBuffersPtr);
    }

    const auto finished = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::max(1e-9, std::chrono::duration<double>(finished - started).count());
    const double rate = toDouble(searched) / elapsedSeconds;
    std::cerr << '\n' << "Hash rate: " << formatHashRate(rate) << '\n';
    return 0;
}