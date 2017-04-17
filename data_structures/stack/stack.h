#ifndef _STACK_H
#define _STACK_H

/*
    stack.h
    An implementation of a templated stack. Supports automatic resizing
    when the size is low.

*/

<template T>
class StackArray
{
    public:
        // Constructor.
        Stack(int size);

        // Destructor.
        ~Stack();

        // copy Constructor
        //Stack Stack(const Stack & s);

        // add to the Stack
        void push(const T &data);

        // pop off the top of the stack
        T pop();

        // peek at the top of the stack
        T peek();

        // isEmpty. Check if stack empty
        bool isEmpty();

    private:
        // array to store stack
        T* array;

        // head pointer
        int head;

        // size
        int size;

};

<template T>
class StackLL
{
    public:
        // constructor
        Stack(int size);

        // Destructor
        ~Stack();

        // copy Constructor
        //Stack Stack(const Stack & s);

        // add to the Stack
        void push(const T &obj);

        // required for abstract class
        void insert(const T &data);

        // required for abstract class
        T remove();

        // pop off the top of the stack
        T pop();

        // peek at the top of the stack
        T peek();

        // isEmpty. Check if stack is empty
        bool is_empty();

    private:
        class Node
        {
            Node(T data)
            {
                this.data = data;
            };

            

            Node* next;
            T data;
        };

        // LL

        // head pointer
        T* head;

        // size
        int size;

};

#endif
