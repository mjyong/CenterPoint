#include "seres_vector.h"
#include <gtest/gtest.h>

class VectorTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};

TEST_F(VectorTest, PushFront)
{
    Seres::vector<int, 64> test_vec;

    test_vec.push_front(2);
    test_vec.push_front(1);
    test_vec.push_front(0);

    EXPECT_EQ(test_vec[0], 0);
    EXPECT_EQ(test_vec[1], 1);
    EXPECT_EQ(test_vec[2], 2);
}

TEST_F(VectorTest, PushBack)
{
    Seres::vector<int, 64> test_vec;

    test_vec.push_back(2);
    test_vec.push_back(1);
    test_vec.push_back(0);

    EXPECT_EQ(test_vec[0], 2);
    EXPECT_EQ(test_vec[1], 1);
    EXPECT_EQ(test_vec[2], 0);
}

TEST_F(VectorTest, Data)
{
    Seres::vector<int, 64> test_vec;

    test_vec.push_back(2);
    test_vec.push_back(1);
    test_vec.push_back(0);

    EXPECT_EQ(test_vec.data()[0], 2);
    EXPECT_EQ(test_vec.data()[1], 1);
    EXPECT_EQ(test_vec.data()[2], 0);
}

TEST_F(VectorTest, Resize)
{
    Seres::vector<int, 64> test_vec;

    test_vec.push_back(2);
    test_vec.push_back(1);
    test_vec.push_back(0);

    EXPECT_EQ(test_vec.size(), 3);

    test_vec.resize(5);
    EXPECT_EQ(test_vec.size(), 5);
    EXPECT_EQ(test_vec[4], 0);

    test_vec.resize(1);
    EXPECT_EQ(test_vec.size(), 1);
    EXPECT_EQ(test_vec.data()[1], 0);
}

TEST_F(VectorTest, Clear)
{
    Seres::vector<int, 64> test_vec;

    test_vec.push_back(2);
    test_vec.push_back(1);
    test_vec.push_back(0);

    test_vec.clear();

    EXPECT_EQ(test_vec.size(), 0);
    EXPECT_EQ(test_vec.data()[0], 0);
}

TEST_F(VectorTest, ForView)
{
    Seres::vector<int, 64> test_vec;

    test_vec.push_back(0);
    test_vec.push_back(1);
    test_vec.push_back(2);

    int i = 0;
    for (auto &value : test_vec)
    {
        EXPECT_EQ(value, i++);
    }
}