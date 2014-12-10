
#ifndef VIDEO_DETECTION_V8_NONCOPYABLE
#define VIDEO_DETECTION_V8_NONCOPYABLE

namespace vd
{
    class nonCopyable
    {
    public:
        nonCopyable() = default;
        nonCopyable(nonCopyable &) = delete;
        nonCopyable & operator=(const nonCopyable &) = delete;
    };
}
#endif