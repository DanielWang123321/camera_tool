#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include <xv-sdk.h>

namespace {
std::shared_ptr<xv::Device> device;
std::shared_ptr<xv::ColorCamera> camera;
std::shared_ptr<xv::CameraModel> model;
int camera_index = 1;
std::string error, info;

std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (const auto c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) >= 32) out += c;
    }
    return out + "\"";
}
}

extern "C" const char* calib_error() { return error.c_str(); }
extern "C" const char* calib_info() { return info.c_str(); }

extern "C" int calib_open(const char* serial, int index) {
    try {
        xv::setLogLevel(xv::LogLevel::warn);
        auto devices = xv::getDevices(5.0);
        if (devices.empty()) throw std::runtime_error("No wrist camera device detected");
        const bool has_serial = serial && serial[0] != '\0';
        auto it = devices.end();
        if (has_serial) {
            it = devices.find(serial);
            if (it == devices.end()) throw std::runtime_error("Specified wrist camera serial not found");
        } else {
            if (devices.size() != 1) throw std::runtime_error("Multiple wrist camera devices detected; specify serial");
            it = devices.begin();
        }
        device = it->second;
        if (has_serial && device->id() != serial) throw std::runtime_error("Wrist camera serial mismatch");
        camera = device->colorCamera();
        if (!camera) throw std::runtime_error("Device has no color camera interface");
        if (index != 1 && index != 2) throw std::runtime_error("Color camera index must be 1 or 2");
        camera_index = index;
        const auto& calibrations = camera->calibration();
        std::ostringstream out;
        out << "{\"serial\":" << quoted(device->id())
            << ",\"sdk_version\":" << quoted(xv::version().toString())
            << ",\"camera_index\":" << index << ",\"sensor_count\":" << calibrations.size()
            << ",\"models\":[";
        bool first = true;
        if (calibrations.size() >= static_cast<size_t>(index)) {
            for (const auto& item : calibrations[index - 1].camerasModel) {
                if (!item) continue;
                if (!first) out << ',';
                first = false;
                out << "{\"width\":" << item->width() << ",\"height\":" << item->height()
                    << ",\"sdk_model_type\":" << quoted(typeid(*item).name()) << '}';
            }
        }
        out << "]}";
        info = out.str();
        if (first) throw std::runtime_error("Device did not return CameraModel for specified color sensor");
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
        return -1;
    }
}

extern "C" int calib_model(int width, int height) {
    try {
        const auto& calibrations = camera->calibration();
        if (calibrations.size() < static_cast<size_t>(camera_index))
            throw std::runtime_error("Device did not return calibration parameters for this color sensor");
        const auto& calibration = calibrations[camera_index - 1];
        std::vector<std::shared_ptr<xv::CameraModel>> matches;
        for (const auto& item : calibration.camerasModel)
            if (item && item->width() == width && item->height() == height) matches.push_back(item);
        if (matches.size() != 1)
            throw std::runtime_error("CameraModel matching image dimensions is missing or ambiguous");
        model = matches.front();
        std::ostringstream out;
        out.precision(17);
        out << "{\"serial\":" << quoted(device->id())
            << ",\"sdk_version\":" << quoted(xv::version().toString())
            << ",\"camera_index\":" << camera_index
            << ",\"raw_width\":" << width << ",\"raw_height\":" << height
            << ",\"sdk_model_type\":" << quoted(typeid(*model).name())
            << ",\"seucm_supported\":" << (camera->checkSeucmSupport() ? "true" : "false")
            << ",\"source\":\"device.colorCamera.calibration.camerasModel\"}";
        info = out.str();
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
        return -1;
    }
}

extern "C" int calib_map(int width, int height, const double* k, float* xy) {
    if (!model) return -1;
    int valid = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double ray[3] = {(x - k[2]) / k[0], (y - k[3]) / k[1], 1.0};
            double p[2], back[3];
            bool ok = model->project(ray, p) && std::isfinite(p[0]) && std::isfinite(p[1]) &&
                      p[0] >= 1 && p[1] >= 1 && p[0] < model->width() - 2 && p[1] < model->height() - 2;
            if (ok) {
                ok = model->raytrace(p, back) && back[2] > 0 &&
                     std::hypot((back[0] / back[2] - ray[0]) * k[0], (back[1] / back[2] - ray[1]) * k[1]) < 0.05;
            }
            const int i = 2 * (y * width + x);
            xy[i] = ok ? static_cast<float>(p[0]) : -1.f;
            xy[i + 1] = ok ? static_cast<float>(p[1]) : -1.f;
            if (ok) ++valid;
        }
    }
    return valid;
}

extern "C" void calib_close() {
    model.reset();
    camera.reset();
    device.reset();
}
