#ifndef __QUADTREE_H__
#define __QUADTREE_H__

#include <iostream>

/*
    A quadtree example. Here, eahc quadtree leaf contains a number,
    and each non-leaf contains the average of the leaves under it.
    From there, we can find the nearest neighbor to any single number
    in the tree.

*/

class Pixel
{
    uint8_t red;
    uint8_t blue;
    uint8_t green;
    uint8_t alpha;
}

class Image
{
    public:
        Image();

        ~Image();

        Pixel getPixelAtLocation(int x, int y);

        

}



class Quadtree
{

    public:

        Quadtree();

        ~Quadtree();

        insert(double) data);

        remove(double data);

        findNearest(double data);

        prune();

    private:


        class Node
        {
            public:
                Node ne;
                Node nw;
                Node se;
                Node sw;
                double data;
        };

}
