#pragma once
#include "msl_prelude.h"
// Independently authored declarations for mesh functions (MSL section 5.4 and
// the mesh type of section 2.19). A [[mesh]] entry is dispatched as a grid of
// threadgroups like a kernel, and writes a fixed-capacity set of vertices and
// primitives through a mesh<V, P, NV, NP, T> parameter instead of returning
// anything. The compiler recognizes the methods by name on this owned record;
// there are no bodies here.
namespace metal {
namespace topology {
struct point {};
struct line {};
struct triangle {};
}
// V is the per-vertex struct (a [[position]] field and varyings, exactly as a
// vertex entry returns), P the per-primitive struct, NV and NP the maximum
// vertices and primitives one threadgroup may produce, and T the topology.
template <typename V, typename P, unsigned NV, unsigned NP, typename T>
struct mesh {
  // How many primitives this threadgroup produces; called once, before the
  // vertices and indices it counts are written.
  void set_primitive_count(uint count);
  // Vertex `index` of this threadgroup's output, below NV. The parameters avoid
  // the names `vertex` and `fragment`, which are keywords in Metal's language.
  void set_vertex(uint index, V value);
  // Per-primitive data for primitive `index`, below NP.
  void set_primitive(uint index, P value);
  // One index of the primitive index buffer: `index` counts indices, not
  // primitives, so a triangle topology takes three per primitive, and `ordinal`
  // names one of this threadgroup's vertices.
  void set_index(uint index, uchar ordinal);
};
// The dispatch an object (task) entry launches (MSL 5.4): the object function
// takes one of these and says how many mesh threadgroups its payload feeds.
struct mesh_grid_properties {
  // Called once per object threadgroup, in threadgroup-uniform control flow.
  void set_threadgroups_per_grid(uint3 size);
};
}
