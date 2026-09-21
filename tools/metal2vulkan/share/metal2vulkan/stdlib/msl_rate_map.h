#pragma once
// Independently authored rasterization rate map interface (MSL 6.15, docs/RATE_MAPS.md). Metal's map
// data is opaque; this compiler's is the layout below, which the host writes (m2v::host::rateMapWords)
// and the decoder, ordinary MSL compiled like any helper, reads: a layer is a grid of cells whose
// per-column and per-row quality values in (0, 1] scale screen distance into physical distance.
namespace metal {
struct rasterization_rate_map_data {
  uint layer_count;          // 1 to 4 layers.
  uint reserved[3];
  uint2 screen_size[4];      // Per layer: the screen extent in pixels.
  uint2 physical_size[4];    // Per layer: the physical extent the host rendered (the sums below, rounded up).
  uint columns[4];           // Per layer: horizontal cells, 1 to 16, each screen_size.x / columns wide.
  uint rows[4];              // Per layer: vertical cells, 1 to 16.
  float horizontal[64];      // Per layer (16 per layer): each column's quality in (0, 1].
  float vertical[64];        // Per layer (16 per layer): each row's quality.
};
struct rasterization_rate_map_decoder {
  rasterization_rate_map_data data;
  explicit rasterization_rate_map_decoder(constant rasterization_rate_map_data &source) : data(source) {}
  // Screen to physical along one axis: whole cells before the coordinate contribute width * quality,
  // the cell holding it its fraction * quality.
  float forward(float coordinate, float extent, uint cells, uint first, bool horizontal_axis) const {
    float width = extent / float(cells);
    float physical = 0.0f;
    for (uint cell = 0u; cell < cells; ++cell) {
      float start = float(cell) * width;
      float quality = data.vertical[first + cell];
      if (horizontal_axis) quality = data.horizontal[first + cell];
      if (coordinate >= start + width) physical += width * quality;
      else if (coordinate > start) physical += (coordinate - start) * quality;
    }
    return physical;
  }
  // Physical to screen: the inverse, cell by cell.
  float inverse(float physical, float extent, uint cells, uint first, bool horizontal_axis) const {
    float width = extent / float(cells);
    float consumed = 0.0f;
    for (uint cell = 0u; cell < cells; ++cell) {
      float quality = data.vertical[first + cell];
      if (horizontal_axis) quality = data.horizontal[first + cell];
      float span = width * quality;
      if (physical <= consumed + span) return float(cell) * width + (physical - consumed) / quality;
      consumed += span;
    }
    return extent;
  }
  float2 map_screen_to_physical_coordinates(float2 screen, uint layer_index = 0) const {
    uint layer = layer_index < data.layer_count ? layer_index : 0u;
    return float2(forward(screen.x, float(data.screen_size[layer].x), data.columns[layer], layer * 16u, true),
                  forward(screen.y, float(data.screen_size[layer].y), data.rows[layer], layer * 16u, false));
  }
  uint2 map_screen_to_physical_coordinates(uint2 screen, uint layer_index = 0) const {
    return uint2(map_screen_to_physical_coordinates(float2(screen), layer_index));
  }
  float2 map_physical_to_screen_coordinates(float2 physical, uint layer_index = 0) const {
    uint layer = layer_index < data.layer_count ? layer_index : 0u;
    return float2(inverse(physical.x, float(data.screen_size[layer].x), data.columns[layer], layer * 16u, true),
                  inverse(physical.y, float(data.screen_size[layer].y), data.rows[layer], layer * 16u, false));
  }
  uint2 map_physical_to_screen_coordinates(uint2 physical, uint layer_index = 0) const {
    return uint2(map_physical_to_screen_coordinates(float2(physical), layer_index));
  }
};
}
