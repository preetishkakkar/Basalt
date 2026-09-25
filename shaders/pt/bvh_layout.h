// The software BVH layouts, shared by the builders, the traversal and the cost kernels.
// Include after shared/prelude.h (MSL) or pt/shim.h (C++).
//
// Binary node: four float4, the bounds of both children, so one fetch tests two boxes.
//   n0 = left.min,  w = left data      n1 = left.max,  w = left count
//   n2 = right.min, w = right data     n3 = right.max, w = right count
// A count of zero is an interior child whose data is its node index; kBvhEmpty data marks
// a missing child. A positive count is a leaf: at the top level its data is an instance
// (count 1); at the bottom level its data is the first of `count` triangles.
// Triangle: three float4, the object-space vertices; the first w holds the triangle's index
// within its primitive, as an int's bits.
#pragma once

PT_CONSTANT uint kBvhEmpty = 0xFFFFFFFFu;
PT_CONSTANT uint kBvhLeafTag = 0x80000000u;   // stack entries: a leaf, not a node
PT_CONSTANT uint kBvhCountShift = 27u;         // bottom-level leaf entries: count above the first triangle
PT_CONSTANT uint kBvhFirstMask = 0x07FFFFFFu;
#define PT_BVH_STACK 64
// The quantized wide traversal pushes every hit child, so a wide tree's worst-case stack
// (maximumTraversalStack, every child hit) grows by up to width - 1 per level; LBVH trees reach
// the 60s as BVH8 (Sponza, animated). Wide kernels come in two stack sizes: 64 entries (the
// default kernels; larger local arrays leave the NVIDIA compiler's faster placement, 20%
// slower on Sponza) and PT_WIDE_BVH_STACK_DEEP (the *_deep kernels, for trees whose bound
// exceeds 64). A kernel defines PT_WIDE_BVH_STACK before its includes to choose.
#define PT_WIDE_BVH_STACK_DEEP 96
#ifndef PT_WIDE_BVH_STACK
#define PT_WIDE_BVH_STACK 64
#endif
PT_CONSTANT uint kWideSlotMask = 0xFFu;          // origin.w: slots in use
PT_CONSTANT uint kWideOctantOrdered = 0x100u;    // origin.w: slots assigned by octant

// Quantized wide node (BVH4/8): origin.w = slot count (low byte) and flags; child low/high =
// origin + scale * (word & 0xFFFF) / (word >> 16) per axis; children[i].w = node index, leaf
// entry, or kBvhEmpty for an empty slot. With kWideOctantOrdered, slot s holds the child a ray
// whose direction is negative on the axes of s's bits (x 1, y 2, z 4) meets first, so visiting
// slots k ^ octant for k = 0, 1, ... goes roughly near to far without sorting (Ylitie, Karras
// and Laine 2017); such a node has 8 slots and may leave some empty.
struct PtWideNode {
  float4 origin;
  float4 scale;
  uint4 children[8];
};
