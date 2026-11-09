#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rte_interface/msg/nav_ptz_take_photo_ctrl_multi.hpp>

#include "../include/datapool.h"
#include "../include/logging.h"
#include "../include/ptz_data_detection.h"

namespace seres {
namespace perception {

class PtzDataDetectionTestPeer {
public:
    static void Prepare(PtzDataDetection& det, int wait_count)
    {
        det.param.use_detection = false;
        det.param.save_img = false;
        det.param.wait_count = wait_count;
    }

    static rclcpp::Node::SharedPtr node(PtzDataDetection& det) { return det.node_; }

    static void SetMultiCtrlPublisher(
        PtzDataDetection& det,
        const rclcpp::Publisher<rte_interface::msg::NavPtzTakePhotoCtrlMulti>::SharedPtr& pub)
    {
        det.pub_take_photo_ctrl_multi_ = pub;
    }

    static void SetMultiPhoto(PtzDataDetection& det, int num, int interval_ms)
    {
        det.param.multi_photo_num = num;
        det.param.multi_photo_interval_ms = interval_ms;
    }

    static void OnPhoto(
        PtzDataDetection& det, const rte_interface::msg::NavPtzTakePhotoData::SharedPtr& msg)
    {
        det.TakePhotoDataCallback(msg);
    }

    static void OnMatrix(
        PtzDataDetection& det, const rte_interface::msg::CamTemMatrix::SharedPtr& msg)
    {
        det.CamTemMatrixCallback(msg);
    }

    static void OnDetect(
        PtzDataDetection& det,
        const perception_interface::srv::PerceptionEvent::Request::SharedPtr& request,
        const perception_interface::srv::PerceptionEvent::Response::SharedPtr& response)
    {
        det.HandlePerceptionDetect(request, response);
    }
};

}  // namespace perception
}  // namespace seres

namespace {

using seres::perception::DataPool;
using seres::perception::PtzDataDetection;
using seres::perception::PtzDataDetectionTestPeer;
using seres::perception::PtzLogger;

class RclcppEnvironment : public ::testing::Environment {
public:
    void SetUp() override
    {
        PtzLogger::Init("/tmp/ptz_data_detection_logs", "ptz_take_photo_test");
        if (!rclcpp::ok()) {
            rclcpp::init(0, nullptr);
        }
    }

    void TearDown() override
    {
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
        PtzLogger::Shutdown();
    }
};

::testing::Environment* const kRclcppEnv =
    ::testing::AddGlobalTestEnvironment(new RclcppEnvironment);

sensor_msgs::msg::CompressedImage MakeJpeg(int width, int height, const cv::Scalar& color)
{
    cv::Mat image(height, width, CV_8UC3, color);
    std::vector<uint8_t> encoded;
    EXPECT_TRUE(cv::imencode(".jpg", image, encoded));
    sensor_msgs::msg::CompressedImage msg;
    msg.format = "jpeg";
    msg.data = encoded;
    return msg;
}

rte_interface::msg::NavPtzTakePhotoData::SharedPtr MakePhoto(
    uint32_t channel_id, std::vector<sensor_msgs::msg::CompressedImage> images)
{
    auto msg = std::make_shared<rte_interface::msg::NavPtzTakePhotoData>();
    msg->take_photo_id = channel_id;
    msg->num = static_cast<uint32_t>(images.size());
    msg->images = std::move(images);
    return msg;
}

void ClearCaches()
{
    auto* dp = DataPool::Instance();
    dp->main_data_.grab_rgb_cache.Clear();
    dp->main_data_.grab_infrared_cache.Clear();
    dp->main_data_.matrix_cache.Clear();
}

class TakePhotoTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        det_ = std::make_unique<PtzDataDetection>();
        PtzDataDetectionTestPeer::Prepare(*det_, 30);
        ClearCaches();
    }

    void TearDown() override
    {
        ClearCaches();
        det_.reset();
    }

    std::unique_ptr<PtzDataDetection> det_;
};

TEST_F(TakePhotoTest, IgnoresUnknownChannel)
{
    auto msg = MakePhoto(4, {MakeJpeg(8, 8, cv::Scalar(0, 0, 255))});
    PtzDataDetectionTestPeer::OnPhoto(*det_, msg);
    auto* dp = DataPool::Instance();
    EXPECT_EQ(dp->main_data_.grab_rgb_cache.GetDataSize(), 0);
    EXPECT_EQ(dp->main_data_.grab_infrared_cache.GetDataSize(), 0);
}

TEST_F(TakePhotoTest, EmptyImagesAreDropped)
{
    auto msg = MakePhoto(5, {});
    PtzDataDetectionTestPeer::OnPhoto(*det_, msg);
    EXPECT_EQ(DataPool::Instance()->main_data_.grab_rgb_cache.GetDataSize(), 0);
}

TEST_F(TakePhotoTest, InvalidJpegIsDropped)
{
    sensor_msgs::msg::CompressedImage bad;
    bad.format = "jpeg";
    bad.data = {1, 2, 3, 4};
    PtzDataDetectionTestPeer::OnPhoto(*det_, MakePhoto(5, {bad}));
    EXPECT_EQ(DataPool::Instance()->main_data_.grab_rgb_cache.GetDataSize(), 0);
}

TEST_F(TakePhotoTest, ColorAndInfraredGoToSeparateCaches)
{
    auto color = MakePhoto(5, {MakeJpeg(8, 8, cv::Scalar(0, 0, 255)),
                               MakeJpeg(8, 8, cv::Scalar(0, 255, 0))});
    color->header.stamp.sec = 3;
    color->header.stamp.nanosec = 250000000;
    color->images[0].header.stamp.sec = 3;
    color->images[0].header.stamp.nanosec = 250000000;
    PtzDataDetectionTestPeer::OnPhoto(*det_, color);

    auto infrared = MakePhoto(6, {MakeJpeg(8, 8, cv::Scalar(255, 0, 0))});
    infrared->header.stamp.sec = 9;
    infrared->header.stamp.nanosec = 0;
    PtzDataDetectionTestPeer::OnPhoto(*det_, infrared);

    auto* dp = DataPool::Instance();
    EXPECT_EQ(dp->main_data_.grab_rgb_cache.GetDataSize(), 2);
    EXPECT_EQ(dp->main_data_.grab_infrared_cache.GetDataSize(), 1);

    seres::perception::ImageData rgb;
    ASSERT_EQ(dp->main_data_.grab_rgb_cache.PopNewest(rgb), 0);
    EXPECT_NEAR(rgb.time, 3.25, 1e-6);
    EXPECT_FALSE(rgb.img.empty());

    seres::perception::ImageData ir;
    ASSERT_EQ(dp->main_data_.grab_infrared_cache.PopNewest(ir), 0);
    EXPECT_NEAR(ir.time, 9.0, 1e-6);
}

TEST_F(TakePhotoTest, FallsBackToMessageStamp)
{
    auto msg = MakePhoto(5, {MakeJpeg(4, 4, cv::Scalar(10, 20, 30))});
    msg->header.stamp.sec = 12;
    msg->header.stamp.nanosec = 500000000;
    PtzDataDetectionTestPeer::OnPhoto(*det_, msg);

    seres::perception::ImageData rgb;
    ASSERT_EQ(DataPool::Instance()->main_data_.grab_rgb_cache.PopNewest(rgb), 0);
    EXPECT_NEAR(rgb.time, 12.5, 1e-6);
}

TEST_F(TakePhotoTest, FrozenTemperatureIsSkipped)
{
    auto msg = std::make_shared<rte_interface::msg::CamTemMatrix>();
    msg->is_freezedata = true;
    msg->height = 2;
    msg->width = 2;
    msg->data = {1.f, 2.f, 3.f, 4.f};
    PtzDataDetectionTestPeer::OnMatrix(*det_, msg);
    EXPECT_EQ(DataPool::Instance()->main_data_.matrix_cache.GetDataSize(), 0);
}

TEST_F(TakePhotoTest, TemperatureMatrixIsCached)
{
    auto msg = std::make_shared<rte_interface::msg::CamTemMatrix>();
    msg->is_freezedata = false;
    msg->height = 2;
    msg->width = 3;
    msg->data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    msg->header.stamp.sec = 4;
    PtzDataDetectionTestPeer::OnMatrix(*det_, msg);

    seres::perception::ImageData matrix;
    ASSERT_EQ(DataPool::Instance()->main_data_.matrix_cache.PopNewest(matrix), 0);
    EXPECT_EQ(matrix.img.rows, 2);
    EXPECT_EQ(matrix.img.cols, 3);
    EXPECT_EQ(matrix.img.type(), CV_32F);
    EXPECT_FLOAT_EQ(matrix.img.at<float>(1, 2), 6.f);
    EXPECT_NEAR(matrix.time, 4.0, 1e-6);
}

TEST_F(TakePhotoTest, TruckParkingWithoutPublisherReturnsError)
{
    auto request = std::make_shared<perception_interface::srv::PerceptionEvent::Request>();
    auto response = std::make_shared<perception_interface::srv::PerceptionEvent::Response>();
    request->percept.event_id = perception_interface::msg::PerceptionEvent::TRUCK_PARKING_DETECTION;
    PtzDataDetectionTestPeer::OnDetect(*det_, request, response);

    ASSERT_EQ(response->event_info.size(), 1u);
    EXPECT_EQ(response->event_info.front().object.format, "error");
}

TEST_F(TakePhotoTest, UnsupportedEventReturnsErrorImage)
{
    auto request = std::make_shared<perception_interface::srv::PerceptionEvent::Request>();
    auto response = std::make_shared<perception_interface::srv::PerceptionEvent::Response>();
    request->percept.event_id = 16;
    PtzDataDetectionTestPeer::OnDetect(*det_, request, response);

    ASSERT_EQ(response->event_info.size(), 1u);
    const auto& info = response->event_info.front();
    EXPECT_EQ(info.object.format, "error");
    const std::string text(info.object.data.begin(), info.object.data.end());
    EXPECT_NE(text.find("unsupported event_id: 16"), std::string::npos);
    EXPECT_TRUE(info.box.empty());
}

class PhotoServiceFixture : public TakePhotoTest {
protected:
    void SetUp() override
    {
        TakePhotoTest::SetUp();
        topic_ = "/ptz_unit_test/take_photo_ctrl_" + std::to_string(reinterpret_cast<uintptr_t>(this));
        auto node = PtzDataDetectionTestPeer::node(*det_);
        multi_pub_ = node->create_publisher<rte_interface::msg::NavPtzTakePhotoCtrlMulti>(topic_, 10);
        PtzDataDetectionTestPeer::SetMultiCtrlPublisher(*det_, multi_pub_);

        responder_ = std::make_shared<rclcpp::Node>("ptz_photo_responder");
        multi_sub_ = responder_->create_subscription<rte_interface::msg::NavPtzTakePhotoCtrlMulti>(
            topic_, 10,
            [this](const rte_interface::msg::NavPtzTakePhotoCtrlMulti::SharedPtr msg) {
                {
                    std::lock_guard<std::mutex> lock(multi_mu_);
                    last_multi_ = *msg;
                    got_multi_ = true;
                }
                if (msg->take_photo_id == 6) {
                    auto photo = MakePhoto(6, {MakeJpeg(80, 60, cv::Scalar(40, 40, 40))});
                    photo->header.stamp.sec = 21;
                    PtzDataDetectionTestPeer::OnPhoto(*det_, photo);

                    auto matrix = std::make_shared<rte_interface::msg::CamTemMatrix>();
                    matrix->is_freezedata = false;
                    matrix->height = 4;
                    matrix->width = 5;
                    matrix->data.assign(20, 10.f);
                    matrix->data[2 * 5 + 1] = 80.5f;
                    PtzDataDetectionTestPeer::OnMatrix(*det_, matrix);
                    return;
                }
                const uint32_t count = msg->num > 0 ? msg->num : 1;
                std::vector<sensor_msgs::msg::CompressedImage> images;
                images.reserve(count);
                for (uint32_t i = 0; i < count; ++i) {
                    images.push_back(MakeJpeg(16, 12, cv::Scalar(0, 0, 40 + static_cast<int>(i) * 20)));
                }
                auto photo = MakePhoto(msg->take_photo_id, std::move(images));
                photo->header.stamp.sec = count == 1 ? 20 : 30;
                PtzDataDetectionTestPeer::OnPhoto(*det_, photo);
            });
        executor_.add_node(responder_);
        spin_thread_ = std::thread([this]() { executor_.spin(); });

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (multi_pub_->get_subscription_count() == 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ASSERT_GT(multi_pub_->get_subscription_count(), 0u);
    }

    void TearDown() override
    {
        executor_.cancel();
        if (spin_thread_.joinable()) {
            spin_thread_.join();
        }
        multi_sub_.reset();
        responder_.reset();
        multi_pub_.reset();
        TakePhotoTest::TearDown();
    }

    std::string topic_;
    rclcpp::Publisher<rte_interface::msg::NavPtzTakePhotoCtrlMulti>::SharedPtr multi_pub_;
    rclcpp::Node::SharedPtr responder_;
    rclcpp::Subscription<rte_interface::msg::NavPtzTakePhotoCtrlMulti>::SharedPtr multi_sub_;
    std::mutex multi_mu_;
    rte_interface::msg::NavPtzTakePhotoCtrlMulti last_multi_;
    bool got_multi_ = false;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread spin_thread_;
};

TEST_F(PhotoServiceFixture, RgbEventReturnsJpeg)
{
    auto request = std::make_shared<perception_interface::srv::PerceptionEvent::Request>();
    auto response = std::make_shared<perception_interface::srv::PerceptionEvent::Response>();
    request->percept.event_id = 14;
    PtzDataDetectionTestPeer::OnDetect(*det_, request, response);

    ASSERT_EQ(response->event_info.size(), 1u);
    const auto& info = response->event_info.front();
    EXPECT_EQ(info.object.format, "jpeg");
    EXPECT_FALSE(info.object.data.empty());
    EXPECT_TRUE(info.box.empty());
    cv::Mat raw(1, static_cast<int>(info.object.data.size()), CV_8UC1,
                const_cast<uint8_t*>(info.object.data.data()));
    cv::Mat decoded = cv::imdecode(raw, cv::IMREAD_COLOR);
    EXPECT_FALSE(decoded.empty());
    std::lock_guard<std::mutex> lock(multi_mu_);
    ASSERT_TRUE(got_multi_);
    EXPECT_EQ(last_multi_.take_photo_id, 5u);
    EXPECT_EQ(last_multi_.num, 1u);
}

TEST_F(PhotoServiceFixture, InfraredEventReturnsMaxTemperatureBox)
{
    auto request = std::make_shared<perception_interface::srv::PerceptionEvent::Request>();
    auto response = std::make_shared<perception_interface::srv::PerceptionEvent::Response>();
    request->percept.event_id = 13;
    PtzDataDetectionTestPeer::OnDetect(*det_, request, response);

    ASSERT_EQ(response->event_info.size(), 1u);
    const auto& info = response->event_info.front();
    EXPECT_EQ(info.object.format, "jpeg");
    ASSERT_EQ(info.box.size(), 1u);
    EXPECT_EQ(info.box.front().id, 0u);
    EXPECT_EQ(info.box.front().class_name, "max_temperature");
    EXPECT_FLOAT_EQ(info.box.front().score, 80.5f);
    EXPECT_FLOAT_EQ(info.box.front().center.x, 1.f);
    EXPECT_FLOAT_EQ(info.box.front().center.y, 2.f);
    EXPECT_FLOAT_EQ(info.box.front().size_x, 5.f);
    EXPECT_FLOAT_EQ(info.box.front().size_y, 4.f);
}

TEST_F(PhotoServiceFixture, TruckParkingUsesConfiguredBurst)
{
    PtzDataDetectionTestPeer::SetMultiPhoto(*det_, 3, 40);
    auto request = std::make_shared<perception_interface::srv::PerceptionEvent::Request>();
    auto response = std::make_shared<perception_interface::srv::PerceptionEvent::Response>();
    request->percept.event_id = perception_interface::msg::PerceptionEvent::TRUCK_PARKING_DETECTION;
    PtzDataDetectionTestPeer::OnDetect(*det_, request, response);

    ASSERT_EQ(response->event_info.size(), 3u);
    for (const auto& info : response->event_info) {
        EXPECT_EQ(info.object.format, "jpeg");
        EXPECT_FALSE(info.object.data.empty());
    }
    std::lock_guard<std::mutex> lock(multi_mu_);
    ASSERT_TRUE(got_multi_);
    EXPECT_EQ(last_multi_.take_photo_id, 5u);
    EXPECT_EQ(last_multi_.num, 3u);
    EXPECT_EQ(last_multi_.time_interval, 40u);
}

TEST_F(PhotoServiceFixture, TruckParkingDefaultBurstIsFiveAt20ms)
{
    auto request = std::make_shared<perception_interface::srv::PerceptionEvent::Request>();
    auto response = std::make_shared<perception_interface::srv::PerceptionEvent::Response>();
    request->percept.event_id = perception_interface::msg::PerceptionEvent::TRUCK_PARKING_DETECTION;
    PtzDataDetectionTestPeer::OnDetect(*det_, request, response);

    ASSERT_EQ(response->event_info.size(), 5u);
    std::lock_guard<std::mutex> lock(multi_mu_);
    ASSERT_TRUE(got_multi_);
    EXPECT_EQ(last_multi_.num, 5u);
    EXPECT_EQ(last_multi_.time_interval, 20u);
}

}  // namespace
