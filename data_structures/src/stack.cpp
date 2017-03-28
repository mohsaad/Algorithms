#include "../include/stack.h"

template<class T>
StackArray::Stack(int size)
{
    array = new T[size];
}

// Destructor. Deletes the array.
template<class T>
StackArray::~Stack()
{
    delete [] array;
    size = 0;
}

template<class T>
void StackArray::push(const T &data)
{
    array[head] = T;
    head++;
}

template<class T>
T StackArray::pop();
{
    head--;
    return array[head];
}

template<class T>
T StackArray::peek()
{
    return array[head - 1];
}

template<class T>
bool StackArray::isEmpty()
{
    return size > 0 ? true : false;
}
