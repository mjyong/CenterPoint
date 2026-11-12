#ifndef _SERES_VECTOR_H_
#define _SERES_VECTOR_H_

#include "stddef.h"
#include "stdint.h"
#include <vector>

namespace Seres {
template <typename T, uint32_t N> class vector {
  public:
    using iterator = T *;
    using const_iterator = const T *;

    vector()
    {
    }

    vector(const vector<T, N> &other)
    {
        update(other);
    }

    vector(vector<T, N> &&other)
    {
        update(other);
        other.clear();
    }

    vector<T, N> &operator=(const vector<T, N> &other)
    {
        update(other);
    }

    vector<T, N> &operator=(vector<T, N> &&other)
    {
        update(other);
        other.clear();
    }

    ~vector()
    {
    }

    uint32_t size() const
    {
        return m_size;
    }

    void resize(uint32_t size)
    {
        if (size >= N)
        {
            std::__throw_out_of_range("out of range");
        }
        if (m_size > size)
        {
            for (auto i = size; i < m_size; i++)
            {
                m_data[i] = T();
            }
        }
        else if (m_size < size)
        {
            for (auto i = m_size; i < size; i++)
            {
                m_data[i] = T();
            }
        }
        m_size = size;
    }

    void clear()
    {
        resize(0);
    }

    T *data()
    {
        return m_data;
    }

    T &operator[](uint32_t i)
    {
        if (i >= m_size)
        {
            std::__throw_out_of_range("out of range");
        }

        return m_data[i];
    }

    void push_front(const T &data)
    {
        if (m_size >= N)
        {
            std::__throw_out_of_range("out of range");
        }
        if (m_size > 0)
        {
            for (uint32_t i = m_size; i > 0; i--)
            {
                m_data[i] = std::move(m_data[i - 1]);
            }
        }
        m_size++;
        m_data[0] = data;
    }

    void push_back(const T &data)
    {
        if (m_size >= N)
        {
            std::__throw_out_of_range("out of range");
        }
        m_data[m_size++] = data;
    }

    iterator begin()
    {
        return m_data;
    }
    iterator end()
    {
        return m_data + m_size;
    }

    const_iterator begin() const
    {
        return m_data;
    }
    const_iterator end() const
    {
        return m_data + m_size;
    }

  private:
    void update(const vector<T, N> &other)
    {
        clear();

        for (auto i = 0; i < other.size(); i++)
        {
            m_data[i] = other.m_data[i];
        }

        m_size = other.size();
    }

  private:
    uint32_t m_size = 0;
    T m_data[N];
};
} // namespace Seres

#endif