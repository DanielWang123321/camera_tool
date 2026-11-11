// 只连接彩色相机传感器；模型由设备 SDK 读取，不启动 SLAM 或末端外设。
#include <cstdint>
#include <xv-sdk.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
std::shared_ptr<xv::Device> device;
std::shared_ptr<xv::ColorCamera> camera;
std::shared_ptr<xv::CameraModel> model;
std::mutex frame_mutex;
xv::ColorImage latest;
long long sequence = 0;
int callback_id = -1, camera_index = 1;
bool started_generic = false, started_indexed = false;
std::string error, info;
int video_fd = -1, video_width = 0, video_height = 0;
bool video_streaming = false;
struct VideoBuffer { void* data; size_t size; };
std::vector<VideoBuffer> video_buffers;

int video_ioctl(int fd, unsigned long request, void* argument) {
    int result;
    do { result = ioctl(fd, request, argument); } while (result < 0 && errno == EINTR);
    return result;
}
void require_video_ioctl(unsigned long request, void* argument, const char* operation) {
    if (video_ioctl(video_fd, request, argument) < 0)
        throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (const auto c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) >= 32) out += c;
    }
    return out + "\"";
}
void on_frame(const xv::ColorImage& image) {
    std::lock_guard<std::mutex> guard(frame_mutex);
    latest = image;
    ++sequence;
}
}

extern "C" const char* calib_error() { return error.c_str(); }
extern "C" const char* calib_info() { return info.c_str(); }

// 只枚举格式，不改变设备状态。只接受与 SEUCM 原图尺寸完全一致的 YU12 彩色口。
extern "C" int calib_video_probe(const char* path, int width, int height) {
    const int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) { error = std::strerror(errno); return -1; }
    v4l2_capability cap{};
    bool found = false;
    if (video_ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
        const auto caps = cap.capabilities & V4L2_CAP_DEVICE_CAPS ? cap.device_caps : cap.capabilities;
        if ((caps & V4L2_CAP_VIDEO_CAPTURE) && (caps & V4L2_CAP_STREAMING)) {
            v4l2_frmsizeenum size{};
            size.pixel_format = V4L2_PIX_FMT_YUV420;
            for (size.index = 0; video_ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0; ++size.index) {
                if (size.type == V4L2_FRMSIZE_TYPE_DISCRETE && size.discrete.width == static_cast<unsigned>(width)
                        && size.discrete.height == static_cast<unsigned>(height)) found = true;
            }
        }
    }
    close(fd);
    return found ? 1 : 0;
}

extern "C" int calib_video_start(const char* path, int width, int height) {
    try {
        video_fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (video_fd < 0) throw std::runtime_error(std::strerror(errno));
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = width; format.fmt.pix.height = height;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        require_video_ioctl(VIDIOC_S_FMT, &format, "设置彩色流格式失败");
        if (format.fmt.pix.width != static_cast<unsigned>(width) || format.fmt.pix.height != static_cast<unsigned>(height)
                || format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUV420 || format.fmt.pix.bytesperline != static_cast<unsigned>(width))
            throw std::runtime_error("V4L2 返回的尺寸、格式或行宽与设备内参不匹配");
        video_width = width; video_height = height;
        v4l2_requestbuffers request{};
        request.count = 3; request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; request.memory = V4L2_MEMORY_MMAP;
        require_video_ioctl(VIDIOC_REQBUFS, &request, "申请视频缓冲失败");
        if (request.count < 2 || request.count > 32) throw std::runtime_error("视频缓冲数量无效");
        for (unsigned i = 0; i < request.count; ++i) {
            v4l2_buffer buffer{};
            buffer.type = request.type; buffer.memory = request.memory; buffer.index = i;
            require_video_ioctl(VIDIOC_QUERYBUF, &buffer, "查询视频缓冲失败");
            void* data = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buffer.m.offset);
            if (data == MAP_FAILED) throw std::runtime_error("视频缓冲映射失败");
            video_buffers.push_back({data, buffer.length});
            require_video_ioctl(VIDIOC_QBUF, &buffer, "提交视频缓冲失败");
        }
        auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        require_video_ioctl(VIDIOC_STREAMON, &type, "启动 V4L2 彩色流失败");
        video_streaming = true;
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}

extern "C" long long calib_video_frame(unsigned char* data, int capacity, int* meta,
                                       double* host_ts, long long* stamp_us) {
    try {
        pollfd descriptor{video_fd, POLLIN, 0};
        const int ready = poll(&descriptor, 1, 100);
        if (ready == 0 || (ready < 0 && errno == EINTR)) return 0;
        if (ready < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)))
            throw std::runtime_error("V4L2 视频设备断开或读取失败");
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; buffer.memory = V4L2_MEMORY_MMAP;
        if (video_ioctl(video_fd, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) return 0;
            throw std::runtime_error("V4L2 读取视频缓冲失败");
        }
        if (buffer.index >= video_buffers.size() || buffer.bytesused > static_cast<unsigned>(capacity)
                || buffer.bytesused > video_buffers[buffer.index].size
                || buffer.bytesused != static_cast<unsigned>(video_width * video_height * 3 / 2)
                || (buffer.flags & V4L2_BUF_FLAG_ERROR))
            throw std::runtime_error("V4L2 图像不完整或设备报告坏帧");
        if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) != V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC)
            throw std::runtime_error("V4L2 没有提供单调时钟时间戳，不能与 TCP 同步");
        std::memcpy(data, video_buffers[buffer.index].data, buffer.bytesused);
        meta[0] = video_width; meta[1] = video_height; meta[2] = 1; meta[3] = buffer.bytesused;
        *host_ts = buffer.timestamp.tv_sec + buffer.timestamp.tv_usec * 1e-6;
        *stamp_us = buffer.timestamp.tv_sec * 1000000LL + buffer.timestamp.tv_usec;
        const long long seq = static_cast<long long>(buffer.sequence) + 1;
        require_video_ioctl(VIDIOC_QBUF, &buffer, "归还视频缓冲失败");
        return seq;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}

extern "C" void calib_video_close() {
    if (video_fd >= 0) {
        if (video_streaming) {
            auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            video_ioctl(video_fd, VIDIOC_STREAMOFF, &type);
        }
        for (const auto& buffer : video_buffers) munmap(buffer.data, buffer.size);
        video_buffers.clear();
        close(video_fd); video_fd = -1; video_streaming = false;
    }
}

extern "C" int calib_open(const char* serial, int index) {
    try {
        xv::setLogLevel(xv::LogLevel::warn);
        auto devices = xv::getDevices(5.0);
        if (devices.empty()) throw std::runtime_error("未检测到腕部相机设备");
        const bool has_serial = serial && serial[0] != '\0';
        auto it = devices.end();
        if (has_serial) {
            it = devices.find(serial);
            if (it == devices.end()) throw std::runtime_error("未找到指定腕部相机序列号");
        } else {
            if (devices.size() != 1) throw std::runtime_error("检测到多个腕部相机设备，请指定序列号");
            it = devices.begin();
        }
        device = it->second;
        if (has_serial && device->id() != serial) throw std::runtime_error("腕部相机序列号不一致");
        camera = device->colorCamera();
        if (!camera) throw std::runtime_error("设备没有彩色相机接口");
        if (index != 1 && index != 2) throw std::runtime_error("彩色相机编号必须为 1 或 2");
        camera_index = index;
        const auto& calibrations = camera->calibration();
        std::ostringstream out;
        out << "{\"serial\":" << quoted(device->id())
            << ",\"sdk_version\":" << quoted(xv::version().toString())
            << ",\"camera_index\":" << index << ",\"sensor_count\":" << calibrations.size()
            << ",\"models\":[";
        bool first = true;
        if (calibrations.size() >= static_cast<size_t>(index)) {
            for (const auto& item : calibrations[index-1].camerasModel) {
                if (!item) continue;
                if (!first) out << ',';
                first = false;
                out << "{\"width\":" << item->width() << ",\"height\":" << item->height()
                    << ",\"sdk_model_type\":" << quoted(typeid(*item).name()) << '}';
            }
        }
        out << "]}";
        info = out.str();
        if (first) throw std::runtime_error("设备没有返回指定彩色传感器的 CameraModel，无法可靠处理畸变");
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}

extern "C" int calib_start() {
    try {
        const int index = camera_index;
        callback_id = index == 1 ? camera->registerCallback(on_frame)
                                 : camera->registerCam2Callback(on_frame);
        if (callback_id < 0) throw std::runtime_error("腕部相机图像回调注册失败");
        started_generic = index == 1 && camera->start();
        if (!started_generic) started_indexed = camera->startCameras(index);
        if (!started_generic && !started_indexed)
            throw std::runtime_error("腕部相机彩色流启动失败");
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}

// Python 预分配缓冲；时间戳是实际曝光的 SDK 主机稳态时钟。
extern "C" long long calib_frame(unsigned char* data, int capacity, int* meta, double* host_ts,
                                 long long* edge_ts) {
    std::lock_guard<std::mutex> guard(frame_mutex);
    if (!latest.data || sequence == 0) return 0;
    if (latest.dataSize > static_cast<unsigned int>(capacity)) return -1;
    std::memcpy(data, latest.data.get(), latest.dataSize);
    meta[0] = static_cast<int>(latest.width); meta[1] = static_cast<int>(latest.height);
    meta[2] = static_cast<int>(latest.codec); meta[3] = static_cast<int>(latest.dataSize);
    *host_ts = latest.hostTimestamp; *edge_ts = latest.edgeTimestampUs;
    return sequence;
}

extern "C" int calib_model(int width, int height) {
    try {
        const auto& calibrations = camera->calibration();
        if (calibrations.size() < static_cast<size_t>(camera_index))
            throw std::runtime_error("设备未返回该彩色传感器的标定参数");
        const auto& calibration = calibrations[camera_index - 1];
        std::vector<std::shared_ptr<xv::CameraModel>> matches;
        for (const auto& item : calibration.camerasModel)
            if (item && item->width() == width && item->height() == height) matches.push_back(item);
        if (matches.size() != 1)
            throw std::runtime_error("与实际图像尺寸对应的 CameraModel 缺失或不唯一，停止标定");
        model = matches.front();
        std::ostringstream out;
        out.precision(17);
        out << "{\"serial\":" << quoted(device->id())
            << ",\"sdk_version\":" << quoted(xv::version().toString())
            << ",\"camera_index\":" << camera_index
            << ",\"raw_width\":" << width << ",\"raw_height\":" << height
            << ",\"sdk_model_type\":" << quoted(typeid(*model).name())
            << ",\"seucm_supported\":" << (camera->checkSeucmSupport() ? "true" : "false")
            << ",\"source\":\"device.colorCamera.calibration.camerasModel\""
            << ",\"legacy_pdcm\":[";
        bool first = true;
        for (const auto& p : calibration.pdcm) {
            if (!first) out << ',';
            first = false;
            out << '[' << p.w << ',' << p.h << ',' << p.fx << ',' << p.fy << ',' << p.u0 << ',' << p.v0;
            for (const auto d : p.distor) out << ',' << d;
            out << ']';
        }
        out << "],\"legacy_ucm\":[";
        first = true;
        for (const auto& p : calibration.ucm) {
            if (!first) out << ',';
            first = false;
            out << '[' << p.w << ',' << p.h << ',' << p.fx << ',' << p.fy << ',' << p.u0 << ',' << p.v0 << ',' << p.xi << ']';
        }
        out << "]}";
        info = out.str();
        return 0;
    } catch (const std::exception& e) { error = e.what(); return -1; }
}

// 虚拟针孔像素 -> 原图像素。原模型负责畸变；没有猜测 SEUCM 参数或近似成 PDM。
extern "C" int calib_map(int width, int height, const double* k, float* xy) {
    if (!model) return -1;
    int valid = 0;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const double ray[3] = {(x-k[2])/k[0], (y-k[3])/k[1], 1.0};
        double p[2], back[3];
        bool ok = model->project(ray, p) && std::isfinite(p[0]) && std::isfinite(p[1]) &&
                  p[0] >= 1 && p[1] >= 1 && p[0] < model->width()-2 && p[1] < model->height()-2;
        if (ok) {
            ok = model->raytrace(p, back) && back[2] > 0 &&
                 std::hypot((back[0]/back[2]-ray[0])*k[0], (back[1]/back[2]-ray[1])*k[1]) < 0.05;
        }
        const int i = 2*(y*width+x);
        xy[i] = ok ? static_cast<float>(p[0]) : -1.f;
        xy[i+1] = ok ? static_cast<float>(p[1]) : -1.f;
        if (ok) ++valid;
    }
    return valid;
}

extern "C" void calib_close() {
    if (camera) {
        if (callback_id >= 0) {
            if (camera_index == 1) camera->unregisterCallback(callback_id);
            else camera->unregisterCam2Callback(callback_id);
        }
        if (started_generic) camera->stop();
        if (started_indexed) camera->stopCameras(camera_index);
    }
    model.reset(); camera.reset(); device.reset();
}
