#include "seres_string.h"
#include <gtest/gtest.h>

class StringTest : public ::testing::Test {
protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(StringTest, Create) {
  Seres::string<64> test_str;
  test_str = "abcd";

  EXPECT_EQ(test_str.c_str()[0], 'a');
  EXPECT_EQ(test_str.c_str()[3], 'd');
  EXPECT_EQ(test_str.c_str()[4], '\0');
  EXPECT_EQ(test_str.length(), 4);
}

TEST_F(StringTest, Compare) {
  Seres::string<64> test_str;
  test_str = "abcd";

  Seres::string<64> test_str1;
  test_str1 = std::string("abcd");

  EXPECT_EQ(test_str, "abcd");
  EXPECT_EQ(test_str, std::string("abcd"));
  EXPECT_EQ(test_str, test_str1);

  Seres::string<64> test_str2;
  test_str1 = std::string("abcde");
  EXPECT_NE(test_str, "abcde");
  EXPECT_NE(test_str, std::string("abcde"));
  EXPECT_NE(test_str, test_str2);

  test_str1 = std::string("abc3");
  EXPECT_EQ((test_str == test_str1), false);
}

TEST_F(StringTest, Clear) {
  Seres::string<64> test_str;
  test_str = "abcd";

  EXPECT_EQ(test_str.length(), 4);

  test_str.clear();
  EXPECT_EQ(test_str.length(), 0);
}
