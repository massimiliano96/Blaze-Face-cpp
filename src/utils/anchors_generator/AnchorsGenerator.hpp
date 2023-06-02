#pragma once

#include "utils/anchors/Anchors.hpp"

class AnchorsGenerator {
public:
  static std::vector<Anchor>
  generateAnchors(const SsdAnchorsCalculatorOptions &options);
};