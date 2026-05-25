//
//  cli.cpp
//  MakeHDR
//
//  Standalone HDR merge tool. Shares solver and pipeline logic with the OFX
//  plugin; only the I/O layer (OIIO) and parameter system differ.
//
//  Usage:
//    MakeHDR [OPTIONS] file1 file2 ... -o output.exr
//
//  Options:
//    -o <file>            Output EXR path (default: hdr_output.exr)
//    --times t1 [t2 ...]  Exposure times in seconds, e.g. 1/250 0.5 (override EXIF)
//    --no-calibrate       Use linear response curve instead of Debevec solver
//    --solver <name>      debevec (default) | robertson
//    --samples <n>        Calibration samples per channel (default: 200)
//    --smoothness <f>     Solver smoothness factor (default: 10.0)
//    --depth <n>          Input bit depth: 256 | 1024 | 4096 (default: 256)
//    --exposure <f>       Exposure offset in stops (default: 0.0)
//    --gamma <f>          Output gamma (default: 1.0 = linear EXR)
//    --highlights <f>     Reinhard tone-mapping blend, 0=full 1=off (default: 1.0)
//    --middle-gray <f>    Target middle gray luminance, 0=off (default: 0.18)
//    --verbose            Enable info-level logging
//


#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <cstring>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include "merge.h"


// ---------------------------------------------------------------------------
//  Types
// ---------------------------------------------------------------------------

using Config = makehdr::merge_params;
OIIO_NAMESPACE_USING

/// In-memory image representation (RGBA float).
struct CliImage
{
    std::vector<float> pixels;
    int width      = 0;
    int height     = 0;
    int components = 4;

    CliImage() = default;

    CliImage(int w, int h, int c = 4)
        : width(w), height(h), components(c), pixels(w * h * c, 0.f)
    {
    }

    void* getPixelAddress(int x, int y)
    {
        const int idx = (y * width + x) * components;
        return (idx >= 0 && idx + components <= (int)pixels.size())
            ? (void*)(pixels.data() + idx)
            : nullptr;
    }

    const void* getPixelAddress(int x, int y) const
    {
        const int idx = (y * width + x) * components;
        return (idx >= 0 && idx + components <= (int)pixels.size())
            ? (const void*)(pixels.data() + idx)
            : nullptr;
    }

    float* getPixelData()             { return pixels.data(); }
    const float* getPixelData() const { return pixels.data(); }

    int pixel_size() const { return width * height * components; }
};

/// One exposure bracket: pixel data, exposure time (seconds), and source path.
struct Bracket
{
    std::shared_ptr<CliImage> img;
    float time;
    std::string path;
};

// ---------------------------------------------------------------------------
//  CLI / Parsing
// ---------------------------------------------------------------------------

enum class CliParseResult { ok, help, error };

/// Print usage information to stdout.
static void print_usage(const char* prog, const Config& cfg)
{
    const std::string mg = cfg.use_middle_gray ? std::to_string(cfg.middle_gray) : std::string("off");
    std::cout <<
        "Usage: " << prog << " [OPTIONS] file1 [file2 ...] -o output.exr\n"
        "\n"
        "Options:\n"
        "  -o <file>            Output EXR path (default: hdr_output.exr)\n"
        "  --times t1 [t2 ...]  Exposure times in seconds, e.g. 1/250 (overrides EXIF)\n"
        "  --no-calibrate       Use linear response instead of Debevec solver\n"
        "  --solver <name>      debevec (default) | robertson\n"
        "  --samples <n>        Calibration samples per channel (default: "  << cfg.samples    << ")\n"
        "  --smoothness <f>     Solver smoothness (default: "                << cfg.smoothness << ")\n"
        "  --depth <n>          Input depth: 256 | 1024 | 4096 (default: "   << cfg.depth      << ")\n"
        "  --exposure <f>       Exposure offset in stops (default: "         << cfg.exposure   << ")\n"
        "  --gamma <f>          Output gamma: 1.0=linear EXR (default: "     << cfg.gamma      << ")\n"
        "  --highlights <f>     Reinhard blend 0=full 1=off (default: "      << cfg.highlights << ")\n"
        "  --middle-gray <f>    Target middle gray, 0=off (default: "        << mg             << ")\n"
        "  --colorspace <name>  Output colorspace: linear (default) | acescg\n"
        "  --input-colorspace   Input colorspace: srgb (default) | linear | acescg\n"
        "  --verbose            Enable info-level logging\n"
        "  -h / --help          Show this message\n";
}

static bool require_option_value(const std::string& opt, int idx, int argc)
{
    if (idx + 1 >= argc)
    {
        std::cerr << "ERROR: Missing value for " << opt << "\n";
        return false;
    }
    return true;
}

static bool parse_int_option(const std::string& opt, const char* text, int& out)
{
    try
    {
        size_t pos = 0;
        const std::string s(text);
        int val = std::stoi(s, &pos);
        if (pos != s.size())
        {
            std::cerr << "ERROR: Invalid integer for " << opt << ": '" << s << "'\n";
            return false;
        }
        out = val;
        return true;
    }
    catch (const std::exception&)
    {
        std::cerr << "ERROR: Invalid integer for " << opt << ": '" << text << "'\n";
        return false;
    }
}

static bool parse_float_option(const std::string& opt, const char* text, float& out)
{
    try
    {
        size_t pos = 0;
        const std::string s(text);
        float val = std::stof(s, &pos);
        if (pos != s.size())
        {
            std::cerr << "ERROR: Invalid number for " << opt << ": '" << s << "'\n";
            return false;
        }
        out = val;
        return true;
    }
    catch (const std::exception&)
    {
        std::cerr << "ERROR: Invalid number for " << opt << ": '" << text << "'\n";
        return false;
    }
}

/// Parse a time string like "1/250", "0.004", or "2" into seconds.
static float parse_time(const std::string& s)
{
    const size_t slash = s.find('/');
    if (slash != std::string::npos)
    {
        const float num = std::stof(s.substr(0, slash));
        const float den = std::stof(s.substr(slash + 1));
        if (den == 0.f) throw std::invalid_argument("division by zero in time: " + s);
        return num / den;
    }
    return std::stof(s);
}

static bool parse_times_option(int argc, char* argv[], int& i, std::vector<float>& out)
{
    bool found_time = false;
    while (i + 1 < argc && argv[i + 1][0] != '-')
    {
        try
        {
            const float t = parse_time(argv[++i]);
            if (t <= 0.f)
            {
                std::cerr << "ERROR: --times values must be > 0\n";
                return false;
            }
            out.push_back(t);
            found_time = true;
        }
        catch (const std::exception&)
        {
            --i;
            break;
        }
    }
    if (!found_time)
    {
        std::cerr << "ERROR: --times requires at least one value\n";
        return false;
    }
    return true;
}

static CliParseResult parse_cli_args(int argc, char* argv[],
                                     Config& cfg,
                                     std::string& output,
                                     std::string& colorspace,
                                     std::string& input_colorspace,
                                     std::vector<std::string>& inputs,
                                     std::vector<float>& override_times,
                                     bool& verbose)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];

        if (a == "-h" || a == "--help")
        {
            print_usage(argv[0], cfg);
            return CliParseResult::help;
        }
        else if (a == "-o")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            output = argv[++i];
        }
        else if (a == "--no-calibrate")           { cfg.calibrate = false; }
        else if (a == "--verbose")                { verbose = true; }
        else if (a == "--solver")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            std::string s = argv[++i];
            if (s == "robertson") cfg.solver_type = 1;
            else if (s == "debevec") cfg.solver_type = 0;
            else
            {
                std::cerr << "ERROR: --solver must be 'debevec' or 'robertson'\n";
                return CliParseResult::error;
            }
        }
        else if (a == "--samples")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_int_option(a, argv[++i], cfg.samples)) return CliParseResult::error;
        }
        else if (a == "--smoothness")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_float_option(a, argv[++i], cfg.smoothness)) return CliParseResult::error;
        }
        else if (a == "--depth")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_int_option(a, argv[++i], cfg.depth)) return CliParseResult::error;
        }
        else if (a == "--exposure")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_float_option(a, argv[++i], cfg.exposure)) return CliParseResult::error;
        }
        else if (a == "--gamma")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_float_option(a, argv[++i], cfg.gamma)) return CliParseResult::error;
        }
        else if (a == "--highlights")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_float_option(a, argv[++i], cfg.highlights)) return CliParseResult::error;
        }
        else if (a == "--middle-gray")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            if (!parse_float_option(a, argv[++i], cfg.middle_gray)) return CliParseResult::error;
            cfg.use_middle_gray = (cfg.middle_gray > 0.f);
        }
        else if (a == "--colorspace")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            colorspace = argv[++i];
            if (colorspace != "linear" && colorspace != "acescg")
            {
                std::cerr << "ERROR: --colorspace must be 'linear' or 'acescg'\n";
                return CliParseResult::error;
            }
        }
        else if (a == "--input-colorspace")
        {
            if (!require_option_value(a, i, argc)) return CliParseResult::error;
            input_colorspace = argv[++i];
            if (input_colorspace != "srgb" && input_colorspace != "linear"
                && input_colorspace != "acescg")
            {
                std::cerr << "ERROR: --input-colorspace must be 'srgb', 'linear', or 'acescg'\n";
                return CliParseResult::error;
            }
        }
        else if (a == "--times")
        {
            if (!parse_times_option(argc, argv, i, override_times))
                return CliParseResult::error;
        }
        else if (a[0] != '-')
        {
            inputs.push_back(a);
        }
        else
        {
            std::cerr << "ERROR: Unknown option '" << a << "'\n";
            print_usage(argv[0], cfg);
            return CliParseResult::error;
        }
    }

    return CliParseResult::ok;
}

// ---------------------------------------------------------------------------
//  Validation
// ---------------------------------------------------------------------------

static bool validate_cli_inputs(Config& cfg,
                                const std::vector<std::string>& inputs,
                                const std::string& input_colorspace)
{
    if (cfg.depth != 256 && cfg.depth != 1024 && cfg.depth != 4096)
    {
        std::cerr << "ERROR: --depth must be 256, 1024, or 4096\n";
        return false;
    }
    if (cfg.samples <= 0)
    {
        std::cerr << "ERROR: --samples must be > 0\n";
        return false;
    }
    if (cfg.smoothness < 0.f)
    {
        std::cerr << "ERROR: --smoothness must be >= 0\n";
        return false;
    }
    if (cfg.gamma <= 0.f)
    {
        std::cerr << "ERROR: --gamma must be > 0\n";
        return false;
    }
    if (cfg.highlights < 0.f || cfg.highlights > 1.f)
    {
        std::cerr << "ERROR: --highlights must be in [0, 1]\n";
        return false;
    }
    if (cfg.middle_gray < 0.f)
    {
        std::cerr << "ERROR: --middle-gray must be >= 0\n";
        return false;
    }

    if (inputs.empty())
    {
        std::cerr << "ERROR: No input files specified\n";
        return false;
    }
    if (inputs.size() == 1)
        cfg.calibrate = false;  // calibration needs 2+ exposures

    // Linear/ACEScg inputs are already scene-linear — there is no camera response
    // curve to recover. Calibration would fit a curve to linear data (wrong math).
    if (cfg.calibrate && (input_colorspace == "linear" || input_colorspace == "acescg"))
    {
        std::cerr << "WARNING: --input-colorspace " << input_colorspace
                  << " implies linear inputs; disabling calibration (--no-calibrate).\n"
                  << "         The solver requires gamma-encoded inputs (srgb) to recover"
                  << " a response curve.\n";
        cfg.calibrate = false;
    }

    return true;
}

// ---------------------------------------------------------------------------
//  Loading
// ---------------------------------------------------------------------------

/// Read ExposureTime from OIIO metadata (EXIF rational or float).
static float read_exif_time(const ImageSpec& spec)
{
    // OIIO stores EXIF rationals as TypeRational (two uint32_t: numerator, denominator)
    const ParamValue* pv = spec.find_attribute("ExposureTime");
    if (pv)
    {
        if (pv->type() == TypeDesc(TypeDesc::UINT, 2))
        {
            const unsigned int* r = (const unsigned int*)pv->data();
            if (r[1] != 0) return (float)r[0] / (float)r[1];
        }
        // Fallback: try scalar float representation
        return spec.get_float_attribute("ExposureTime", 0.f);
    }
    return 0.f;
}

/// Load one image into a CliImage (RGBA float).
static std::shared_ptr<CliImage> load_image(ImageBuf& src,
                                             int expected_w, int expected_h)
{
    const ImageSpec& spec = src.spec();
    const int w = spec.width;
    const int h = spec.height;

    if (expected_w > 0 && (w != expected_w || h != expected_h))
    {
        std::cerr << "WARNING: Skipping " << src.name()
                  << " (resolution " << w << "x" << h
                  << " != " << expected_w << "x" << expected_h << ")\n";
        return nullptr;
    }

    auto img = std::make_shared<CliImage>(w, h, 4);

    if (spec.nchannels == 4)
    {
        // Direct read — no intermediate ImageBuf copy
        src.get_pixels(ROI(), TypeFloat, img->pixels.data());
    }
    else
    {
        // Pad/reorder channels to RGBA, then read
        const int map[]           = { 0, 1, 2, -1 };
        const float fill[]        = { 0.f, 0.f, 0.f, 1.f };
        const std::string names[] = { "R", "G", "B", "A" };
        ImageBuf rgba = ImageBufAlgo::channels(src, 4, map, fill,
                            std::vector<std::string>(names, names + 4));
        if (rgba.has_error())
        {
            std::cerr << "ERROR: Channel conversion failed for " << src.name()
                      << ": " << rgba.geterror() << "\n";
            return nullptr;
        }
        rgba.get_pixels(ROI(), TypeFloat, img->pixels.data());
    }

    return img;
}

/// Load all input images and resolve exposure times (EXIF or --times override).
static bool load_entries(const std::vector<std::string>& paths,
                          const std::vector<float>& override_times,
                          int& out_w, int& out_h,
                          bool verbose,
                          std::vector<Bracket>& entries)
{
    for (int i = 0; i < (int)paths.size(); ++i)
    {
        const std::string& path = paths[i];

        // Peek at spec to get resolution and EXIF time without loading pixels yet
        ImageBuf probe(path);
        if (probe.has_error())
        {
            std::cerr << "WARNING: Cannot open " << path << ", skipping\n";
            continue;
        }
        const ImageSpec& spec = probe.spec();
        const int w = spec.width;
        const int h = spec.height;

        if (out_w == 0) { out_w = w; out_h = h; }

        const float t = (i < (int)override_times.size()) ? override_times[i]
                                                          : read_exif_time(spec);
        if (t <= 0.f)
        {
            std::cerr << "ERROR: No exposure time for " << path
                      << ". Use --times to provide exposure times.\n";
            return false;
        }

        auto img = load_image(probe, out_w, out_h);
        if (!img) continue;

        entries.push_back({ img, t, path });
        if (verbose) std::cerr << path << "  " << w << "x" << h << "  t=" << t << "s\n";
    }
    return true;
}

// ---------------------------------------------------------------------------
//  Colorspace
// ---------------------------------------------------------------------------

/// Apply a 3x3 matrix to all RGB pixels in-place; alpha is untouched.
static void apply_matrix(CliImage& img, const float M[3][3])
{
    const int npix = img.width * img.height;
    float* p = img.pixels.data();
    for (int i = 0; i < npix; ++i, p += 4)
    {
        const float r = p[0], g = p[1], b = p[2];
        p[0] = M[0][0]*r + M[0][1]*g + M[0][2]*b;
        p[1] = M[1][0]*r + M[1][1]*g + M[1][2]*b;
        p[2] = M[2][0]*r + M[2][1]*g + M[2][2]*b;
    }
}

/// Convert linear sRGB (Rec.709, D65) → ACEScg (AP1, D60).
/// Matrix from the ACES Common LUT Format specification (Bradford CAT).
static void apply_srgb_to_acescg(CliImage& img)
{
    static const float M[3][3] = {
        {  0.6130974f,  0.3395231f,  0.0473794f },
        {  0.0701242f,  0.9163555f,  0.0135153f },
        {  0.0205877f,  0.1095743f,  0.8698380f }
    };
    apply_matrix(img, M);
}

/// Convert ACEScg (AP1, D60) → linear sRGB (Rec.709, D65).
/// Analytical inverse of the forward matrix.
static void apply_acescg_to_srgb(CliImage& img)
{
    static const float M[3][3] = {   // analytical inverse of the forward matrix
        {  1.7049722f, -0.6217675f, -0.0832078f },
        { -0.1301197f,  1.1407628f, -0.0106373f },
        { -0.0239627f, -0.1289867f,  1.1529488f }
    };
    apply_matrix(img, M);
}

/// Route to the correct colorspace conversion based on input/output tags.
static void apply_colorspace_conversion(CliImage& img,
                                        const std::string& input_cs,
                                        const std::string& output_cs,
                                        bool verbose)
{
    const bool in_aces  = (input_cs == "acescg");
    const bool out_aces = (output_cs == "acescg");
    if (!in_aces && out_aces)
    {
        if (verbose) std::cerr << "converting to ACEScg (AP1) ...\n";
        apply_srgb_to_acescg(img);
    }
    else if (in_aces && !out_aces)
    {
        if (verbose) std::cerr << "converting ACEScg \u2192 linear sRGB ...\n";
        apply_acescg_to_srgb(img);
    }
    // in_aces && out_aces  → already in AP1, just tag correctly (no-op)
    // !in_aces && !out_aces → both Rec.709, no conversion needed
}

// ---------------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------------

static std::string build_exposure_list(const std::vector<float>& exposures)
{
    if (exposures.empty())
        return {};

    std::string exposure_list;
    // Typical float text from std::to_string is short; reserve to reduce reallocations.
    exposure_list.reserve(exposures.size() * 12);

    exposure_list += std::to_string(exposures[0]);
    for (size_t i = 1; i < exposures.size(); ++i)
    {
        exposure_list += ",";
        exposure_list += std::to_string(exposures[i]);
    }
    return exposure_list;
}

/// Save result as EXR (32-bit float, Nuke-friendly scanline layout).
static bool save_exr(const std::string& path, const CliImage& img,
                     const std::string& colorspace,
                     const std::vector<float>& exposures = {},
                     const std::string& compression = "zips")
{
    // Build a 3-channel ImageBuf directly using stride to skip the alpha channel
    ImageSpec out_spec(img.width, img.height, 3, TypeFloat);
    // Supported: none, rle, zip, zips (default), piz, pxr24, b44, b44a, dwaa, dwab
    out_spec.attribute("compression", compression);
    out_spec.channelnames = { "R", "G", "B" };
    out_spec.attribute("PixelAspectRatio", 1.0f);
    out_spec.attribute("oiio:ColorSpace",  colorspace);
    out_spec.attribute("Software", "MakeHDR");

    if (!exposures.empty())
    {
        out_spec.attribute("makehdr:ExposureCount", static_cast<int>(exposures.size()));
        out_spec.attribute("makehdr:ExposureTimes", build_exposure_list(exposures));
        out_spec.attribute("ImageDescription", "Merged HDR compiled from " + std::to_string(exposures.size()) + " exposures");
    }

    ImageBuf rgb(out_spec);
    const stride_t xstride = 4 * sizeof(float);   // 4 floats per pixel, read only R/G/B
    rgb.set_pixels(ROI(), TypeFloat, img.pixels.data(), xstride);

    auto out = ImageOutput::create(path);
    if (!out)
    {
        std::cerr << "ERROR: Cannot create output " << path
                  << ": " << OIIO::geterror() << "\n";
        return false;
    }

    if (!out->open(path, out_spec))
    {
        std::cerr << "ERROR: Cannot open output for writing " << path
                  << ": " << out->geterror() << "\n";
        return false;
    }
    if (!rgb.write(out.get()))
    {
        std::cerr << "ERROR: Write failed for " << path
                  << ": " << rgb.geterror() << "\n";
        out->close();
        return false;
    }
    out->close();
    return true;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    Config cfg;
    cfg.use_middle_gray = true;   // CLI default: on (pass --middle-gray 0 to disable)

    if (argc < 2)
    {
        print_usage(argv[0], cfg);
        return 1;
    }

    // 1. Parse
    std::string output         = "hdr_output.exr";
    std::string colorspace     = "linear";
    std::string input_colorspace = "srgb";
    std::vector<std::string> inputs;
    std::vector<float> override_times;
    bool verbose = false;

    const CliParseResult parse_result = parse_cli_args(
        argc, argv, cfg, output, colorspace, input_colorspace, inputs, override_times, verbose);
    if (parse_result == CliParseResult::help)
        return 0;
    if (parse_result == CliParseResult::error)
        return 1;

    // 2. Validate
    if (!validate_cli_inputs(cfg, inputs, input_colorspace))
        return 1;

    // 3. Load
    int dom_w = 0, dom_h = 0;
    std::vector<Bracket> entries;

    if (!load_entries(inputs, override_times, dom_w, dom_h, verbose, entries))
        return 1;
    if (entries.empty()) { std::cerr << "ERROR: No usable images loaded\n"; return 1; }

    // 4. Sort by exposure time (ascending = darkest first, same as OFX)
    std::sort(entries.begin(), entries.end(),
              [](const Bracket& a, const Bracket& b){ return a.time < b.time; });

    // 5. Build source / time vectors
    std::vector<std::shared_ptr<CliImage>> sources;
    std::vector<float> exp_times;
    std::vector<float> exp_times_log;
    sources.reserve(entries.size());
    exp_times.reserve(entries.size());
    exp_times_log.reserve(entries.size());

    for (const auto& e : entries)
    {
        sources.push_back(e.img);
        exp_times.push_back(e.time);
        exp_times_log.push_back(std::log(e.time));
    }

    // 6. Calibrate
    makehdr::timer timer;

    const std::vector<float> weights = makehdr::build_weights(cfg.depth);

    std::vector<double> response;        // size: depth * 3 (calibrated)
    std::vector<double> response_linear; // size: depth     (linear)

    response_linear.resize(cfg.depth);
    makehdr::build_linear_response(cfg.depth, response_linear.data());

    if (cfg.calibrate)
    {
        response.assign(cfg.depth * CMP_MAX, 0.0);
        if (cfg.solver_type == 0)
        {
            const std::vector<makehdr::point> sample_pts =
                makehdr::build_sample_points(dom_w, dom_h, cfg.samples);
            if (verbose) std::cerr << "calibrating: " << sample_pts.size() << " samples, solver=debevec\n";
            makehdr::run_debevec_calibration<float, CliImage>(
                sources, exp_times_log, cfg, weights, sample_pts, response.data());
        }
        else if (cfg.solver_type == 1)
        {
            const std::vector<makehdr::point> sample_pts =
                makehdr::build_sample_points(dom_w, dom_h, cfg.samples * 100);
            if (verbose) std::cerr << "calibrating: " << sample_pts.size() << " samples, solver=robertson\n";
            makehdr::run_robertson_calibration<float, CliImage>(
                sources, exp_times, cfg, weights, sample_pts, response.data());
        }
    }

    // 7. Merge
    CliImage dst(dom_w, dom_h, 4);
    if (verbose) std::cerr << "merging " << sources.size() << " exposures ...\n";

    makehdr::merge_hdr_region<float, CliImage, CliImage>(
        sources,
        exp_times_log,
        weights,
        response.data(),
        response_linear.data(),
        &dst,
        0, 0, dst.width, dst.height,
        cfg,
        []() { return false; });

    // 8. Post-process
    makehdr::post_process_buffer<float>(
        dst.getPixelData(),
        dst.pixel_size(),
        cfg);

    // 9. Convert colorspace
    apply_colorspace_conversion(dst, input_colorspace, colorspace, verbose);

    if (verbose) std::cerr << "done in " << timer.get() << "ms\n";

    // 10. Save
    auto ends_with_ci = [](const std::string& s, const std::string& suffix) {
        if (s.size() < suffix.size()) return false;
        const std::string tail = s.substr(s.size() - suffix.size());
        std::string tail_lower = tail;
        std::transform(tail_lower.begin(), tail_lower.end(), tail_lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return tail_lower == suffix;
    };
    if (!ends_with_ci(output, ".exr"))
        output += ".exr";

    if (!save_exr(output, dst, colorspace, exp_times))
        return 1;

    std::cout << "Saved: " << output << "  (" << dom_w << "x" << dom_h
              << ", " << sources.size() << " exposures)\n";
    return 0;
}
