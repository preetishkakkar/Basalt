// IEEE 754 binary32 arithmetic on bit patterns: round to nearest even, subnormals kept, NaN
// results as the default quiet NaN. For GPU code that must reproduce a host float computation bit
// for bit: Vulkan allows divisions within 3 ULP and may flush subnormals, so a kernel publishing
// the same output as a CPU oracle (the wide collapse: buildWideBvh) computes on bits: integer
// routines after Berkeley SoftFloat 3's f32 ones, and the device's own add and multiply where
// Vulkan guarantees them exact (all operands and results normal). Shared with the host, whose
// tests compare every operation with the native one.
module pt_exact_float;

// The leading zero bits of x, 32 for zero.
uint leadingZeros(uint x) { return x == 0u ? 32u : 31u - firstbithigh(x); }

// The high word of the 64-bit product, from 16-bit halves (no 64-bit integers needed).
uint mulHigh(uint a, uint b) {
  const uint aLow = a & 0xFFFFu, aHigh = a >> 16u, bLow = b & 0xFFFFu, bHigh = b >> 16u;
  const uint lowLow = aLow * bLow, lowHigh = aLow * bHigh, highLow = aHigh * bLow;
  const uint carry = ((lowLow >> 16u) + (lowHigh & 0xFFFFu) + (highLow & 0xFFFFu)) >> 16u;
  return aHigh * bHigh + (lowHigh >> 16u) + (highLow >> 16u) + carry;
}

public static const uint kExactNaN = 0x7FC00000u;
public static const uint kExactPositiveInfinity = 0x7F800000u;
public static const uint kExactNegativeInfinity = 0xFF800000u;

public bool ptExactIsNaN(uint a) { return (a & 0x7FFFFFFFu) > 0x7F800000u; }

// Sign, biased exponent and fraction. The significand's implicit bit is added by the plus.
public uint ptExactPack(uint sign, int exponent, uint significand) {
  return (sign << 31u) + (uint(exponent) << 23u) + significand;
}

// a >> distance, with any bit shifted out ORed into bit 0.
public uint ptExactShiftRightJam(uint a, uint distance) {
  if (distance == 0u) return a;
  if (distance >= 31u) return a != 0u ? 1u : 0u;
  return (a >> distance) | ((a << (32u - distance)) != 0u ? 1u : 0u);
}

// significand: the value with its leading one at bit 30 (or below for subnormal results) and
// seven rounding bits; exponent: the biased exponent minus one.
public uint ptExactRoundPack(uint sign, int exponent, uint significand) {
  uint roundBits = significand & 0x7Fu;
  if (exponent < 0) {
    significand = ptExactShiftRightJam(significand, uint(-exponent));
    exponent = 0;
    roundBits = significand & 0x7Fu;
  } else if (exponent > 0xFD || (exponent == 0xFD && significand + 0x40u >= 0x80000000u)) {
    return ptExactPack(sign, 0xFF, 0u);
  }
  significand = (significand + 0x40u) >> 7u;
  if (roundBits == 0x40u) significand &= ~1u;
  if (significand == 0u) exponent = 0;
  return ptExactPack(sign, exponent, significand);
}

// ptExactRoundPack for a significand not yet normalised: its leading one moved to bit 30,
// packed directly when the shift leaves the rounding bits clear.
public uint ptExactNormRoundPack(uint sign, int exponent, uint significand) {
  const int shift = int(leadingZeros(significand)) - 1;
  exponent -= shift;
  if (shift >= 7 && exponent >= 0 && exponent < 0xFD) {
    if (significand == 0u) exponent = 0;
    return ptExactPack(sign, exponent, significand << uint(shift - 7));
  }
  return ptExactRoundPack(sign, exponent, significand << uint(shift));
}

public int ptExactExponent(uint a) { return int((a >> 23u) & 0xFFu); }

// Operands of the same sign.
public uint ptExactAddMagnitudes(uint a, uint b) {
  const uint sign = a >> 31u;
  const int expA = ptExactExponent(a), expB = ptExactExponent(b);
  uint sigA = a & 0x7FFFFFu, sigB = b & 0x7FFFFFu;
  const int difference = expA - expB;
  int expZ = 0;
  uint sigZ = 0u;
  if (difference == 0) {
    // Two subnormals add as integers; a carry into bit 23 gives the right normal.
    if (expA == 0) return a + sigB;
    if (expA == 0xFF) return (sigA | sigB) != 0u ? kExactNaN : a;
    expZ = expA;
    sigZ = 0x01000000u + sigA + sigB;
    // An even sum is exact: pack it without rounding.
    if ((sigZ & 1u) == 0u && expZ < 0xFE) return ptExactPack(sign, expZ, sigZ >> 1u);
    sigZ <<= 6u;
  } else {
    sigA <<= 6u;
    sigB <<= 6u;
    if (difference < 0) {
      if (expB == 0xFF) return sigB != 0u ? kExactNaN : ptExactPack(sign, 0xFF, 0u);
      expZ = expB;
      if (expA != 0) sigA += 0x20000000u;
      else sigA += sigA;
      sigA = ptExactShiftRightJam(sigA, uint(-difference));
    } else {
      if (expA == 0xFF) return sigA != 0u ? kExactNaN : a;
      expZ = expA;
      if (expB != 0) sigB += 0x20000000u;
      else sigB += sigB;
      sigB = ptExactShiftRightJam(sigB, uint(difference));
    }
    sigZ = 0x20000000u + sigA + sigB;
    if (sigZ < 0x40000000u) {
      expZ -= 1;
      sigZ <<= 1u;
    }
  }
  return ptExactRoundPack(sign, expZ, sigZ);
}

// Operands of opposite signs: a + b = a - |b| with a's sign.
public uint ptExactSubtractMagnitudes(uint a, uint b) {
  uint sign = a >> 31u;
  int expA = ptExactExponent(a);
  const int expB = ptExactExponent(b);
  uint sigA = a & 0x7FFFFFu, sigB = b & 0x7FFFFFu;
  const int difference = expA - expB;
  if (difference == 0) {
    if (expA == 0xFF) return kExactNaN;
    int sigDifference = int(sigA) - int(sigB);
    if (sigDifference == 0) return 0u;
    if (expA != 0) expA -= 1;
    if (sigDifference < 0) {
      sign ^= 1u;
      sigDifference = -sigDifference;
    }
    int shift = int(leadingZeros(uint(sigDifference))) - 8;
    int expZ = expA - shift;
    if (expZ < 0) {
      shift = expA;
      expZ = 0;
    }
    return ptExactPack(sign, expZ, uint(sigDifference) << uint(shift));
  }
  sigA <<= 7u;
  sigB <<= 7u;
  int expZ = 0;
  uint sigX = 0u, sigY = 0u, distance = 0u;
  if (difference < 0) {
    sign ^= 1u;
    if (expB == 0xFF) return sigB != 0u ? kExactNaN : ptExactPack(sign, 0xFF, 0u);
    expZ = expB - 1;
    sigX = sigB | 0x40000000u;
    sigY = sigA;
    if (expA != 0) sigY += 0x40000000u;
    else sigY += sigA;
    distance = uint(-difference);
  } else {
    if (expA == 0xFF) return sigA != 0u ? kExactNaN : a;
    expZ = expA - 1;
    sigX = sigA | 0x40000000u;
    sigY = sigB;
    if (expB != 0) sigY += 0x40000000u;
    else sigY += sigB;
    distance = uint(difference);
  }
  return ptExactNormRoundPack(sign, expZ, sigX - ptExactShiftRightJam(sigY, distance));
}

// A normal float: neither zero, subnormal, infinite nor NaN.
public bool ptExactNormal(uint a) {
  const uint exponent = (a >> 23u) & 0xFFu;
  return exponent != 0u && exponent != 0xFFu;
}

// Vulkan rounds float add, subtract and multiply correctly (-fp-mode precise marks them NoContraction);
// only zeros, subnormals, infinities and NaN may differ. So the device's result is the IEEE one
// whenever both operands and the result are normal, and the bit routines handle the rest. Zero
// operands and exact cancellations, common with flat boxes, are answered directly.
public uint ptExactAdd(uint a, uint b) {
  const uint magnitudeA = a & 0x7FFFFFFFu, magnitudeB = b & 0x7FFFFFFFu;
  if (magnitudeA == 0u) {
    if (magnitudeB == 0u) return a & b;  // -0 only from -0 + -0
    return b;
  }
  if (magnitudeB == 0u) return a;
  if (ptExactNormal(a) && ptExactNormal(b)) {
    if (a == (b ^ 0x80000000u)) return 0u;  // x + -x = +0
    const uint sum = asuint(asfloat(a) + asfloat(b));
    if (ptExactNormal(sum)) return sum;
  }
  if (((a ^ b) >> 31u) != 0u) return ptExactSubtractMagnitudes(a, b);
  return ptExactAddMagnitudes(a, b);
}

public uint ptExactSubtract(uint a, uint b) { return ptExactAdd(a, b ^ 0x80000000u); }

// A subnormal's fraction, normalised: its leading one at bit 23 and the matching exponent.
public void ptExactNormalise(inout int exponent, inout uint significand) {
  const int shift = int(leadingZeros(significand)) - 8;
  exponent = 1 - shift;
  significand <<= uint(shift);
}

public uint ptExactMultiply(uint a, uint b) {
  // A zero times a finite value: a zero with the product's sign.
  if (((a & 0x7FFFFFFFu) == 0u && ptExactExponent(b) != 0xFF) ||
      ((b & 0x7FFFFFFFu) == 0u && ptExactExponent(a) != 0xFF))
    return (a ^ b) & 0x80000000u;
  if (ptExactNormal(a) && ptExactNormal(b)) {
    const uint product = asuint(asfloat(a) * asfloat(b));
    if (ptExactNormal(product)) return product;
  }
  const uint sign = (a ^ b) >> 31u;
  int expA = ptExactExponent(a), expB = ptExactExponent(b);
  uint sigA = a & 0x7FFFFFu, sigB = b & 0x7FFFFFu;
  if (expA == 0xFF) {
    if (sigA != 0u || (expB == 0xFF && sigB != 0u)) return kExactNaN;
    if ((uint(expB) | sigB) == 0u) return kExactNaN;
    return ptExactPack(sign, 0xFF, 0u);
  }
  if (expB == 0xFF) {
    if (sigB != 0u) return kExactNaN;
    if ((uint(expA) | sigA) == 0u) return kExactNaN;
    return ptExactPack(sign, 0xFF, 0u);
  }
  if (expA == 0) {
    if (sigA == 0u) return sign << 31u;
    ptExactNormalise(expA, sigA);
  }
  if (expB == 0) {
    if (sigB == 0u) return sign << 31u;
    ptExactNormalise(expB, sigB);
  }
  int expZ = expA + expB - 0x7F;
  sigA |= 0x800000u;
  sigB |= 0x800000u;
  // The 48-bit product high * 2^24 + low, from 12-bit halves.
  const uint aHigh = sigA >> 12u, aLow = sigA & 0xFFFu, bHigh = sigB >> 12u, bLow = sigB & 0xFFFu;
  const uint middle = aHigh * bLow + aLow * bHigh;
  const uint lowSum = aLow * bLow + ((middle & 0xFFFu) << 12u);
  const uint low = lowSum & 0xFFFFFFu;
  const uint high = aHigh * bHigh + (middle >> 12u) + (lowSum >> 24u);
  uint sigZ = (high << 7u) | (low >> 17u);
  if ((low & 0x1FFFFu) != 0u) sigZ |= 1u;
  if (sigZ < 0x40000000u) {
    expZ -= 1;
    sigZ <<= 1u;
  }
  return ptExactRoundPack(sign, expZ, sigZ);
}

// floor(sigA * 2^bits / sigB) by restoring division, any remainder jammed into bit 0. For
// significands in [2^23, 2^24), with bits 30 when sigA >= sigB, else 31, the quotient's leading
// one lands at bit 30.
public uint ptExactQuotientBitwise(uint sigA, uint sigB, uint bits) {
  uint quotient = 0u, remainder = sigA;
  if (remainder >= sigB) {
    quotient = 1u;
    remainder -= sigB;
  }
  for (uint i = 0u; i < bits; ++i) {
    remainder <<= 1u;
    quotient <<= 1u;
    if (remainder >= sigB) {
      remainder -= sigB;
      quotient |= 1u;
    }
  }
  if (remainder != 0u) quotient |= 1u;
  return quotient;
}

// The same quotient from an estimate (the device's division is within a few hundred units): the
// remainder sigA * 2^bits - q * sigB is exact as a two-word integer, and a float-estimated step
// plus exact unit steps correct q. Exact whatever the estimate; the bitwise routine answers if
// the steps do not settle.
public uint ptExactQuotientFrom(uint sigA, uint sigB, uint bits, uint quotient) {
  const uint numeratorLow = sigA << bits, numeratorHigh = sigA >> (32u - bits);
  uint low = numeratorLow - quotient * sigB;
  int high = int(numeratorHigh - mulHigh(quotient, sigB) - (numeratorLow < quotient * sigB ? 1u : 0u));
  // One step by the estimated remainder / sigB (a few hundred at most).
  const int step = int(floor((float(high) * 4294967296.0f + float(low)) / float(sigB)));
  const uint magnitude = uint(abs(step));
  const uint stepLow = magnitude * sigB, stepHigh = mulHigh(magnitude, sigB);
  if (step > 0) {
    quotient += magnitude;
    high -= int(stepHigh + (low < stepLow ? 1u : 0u));
    low -= stepLow;
  } else if (step < 0) {
    quotient -= magnitude;
    const uint sum = low + stepLow;
    high += int(stepHigh + (sum < stepLow ? 1u : 0u));
    low = sum;
  }
  for (uint i = 0u; i < 4u; ++i) {
    if (high < 0) {
      quotient -= 1u;
      const uint sum = low + sigB;
      high += int(sum < sigB ? 1u : 0u);
      low = sum;
    } else if (high > 0 || low >= sigB) {
      quotient += 1u;
      high -= int(low < sigB ? 1u : 0u);
      low -= sigB;
    } else {
      if (low != 0u) quotient |= 1u;
      return quotient;
    }
  }
  return ptExactQuotientBitwise(sigA, sigB, bits);
}

public uint ptExactQuotient(uint sigA, uint sigB, uint bits) {
  return ptExactQuotientFrom(sigA, sigB, bits, uint(float(sigA) / float(sigB) * float(1u << bits)));
}

public uint ptExactDivide(uint a, uint b) {
  const uint sign = (a ^ b) >> 31u;
  int expA = ptExactExponent(a), expB = ptExactExponent(b);
  uint sigA = a & 0x7FFFFFu, sigB = b & 0x7FFFFFu;
  if (expA == 0xFF) {
    if (sigA != 0u || expB == 0xFF) return kExactNaN;
    return ptExactPack(sign, 0xFF, 0u);
  }
  if (expB == 0xFF) {
    if (sigB != 0u) return kExactNaN;
    return sign << 31u;
  }
  if (expB == 0) {
    if (sigB == 0u) {
      if ((uint(expA) | sigA) == 0u) return kExactNaN;
      return ptExactPack(sign, 0xFF, 0u);
    }
    ptExactNormalise(expB, sigB);
  }
  if (expA == 0) {
    if (sigA == 0u) return sign << 31u;
    ptExactNormalise(expA, sigA);
  }
  int expZ = expA - expB + 0x7E;
  sigA |= 0x800000u;
  sigB |= 0x800000u;
  uint bits = 30u;
  if (sigA < sigB) {
    expZ -= 1;
    bits = 31u;
  }
  return ptExactRoundPack(sign, expZ, ptExactQuotient(sigA, sigB, bits));
}

// a < b as IEEE compares: false with a NaN, and the zeros equal.
public bool ptExactLess(uint a, uint b) {
  if (ptExactIsNaN(a) || ptExactIsNaN(b)) return false;
  const uint signA = a >> 31u, signB = b >> 31u;
  if (signA != signB) return signA != 0u && ((a | b) & 0x7FFFFFFFu) != 0u;
  if (signA == 0u) return a < b;
  return a > b;
}

// The host's pt::min and pt::max: a < b ? a : b and a > b ? a : b.
public uint ptExactMin(uint a, uint b) {
  if (ptExactLess(a, b)) return a;
  return b;
}
public uint ptExactMax(uint a, uint b) {
  if (ptExactLess(b, a)) return a;
  return b;
}

// std::nextafter towards +infinity (up) or -infinity.
public uint ptExactNextAfter(uint a, bool up) {
  if (ptExactIsNaN(a)) return a;
  if ((a & 0x7FFFFFFFu) == 0u) {
    if (up) return 1u;
    return 0x80000001u;
  }
  if (up && a == kExactPositiveInfinity) return a;
  if (!up && a == kExactNegativeInfinity) return a;
  if (((a >> 31u) == 0u) == up) return a + 1u;
  return a - 1u;
}

// uint(clamp(floor(q), 0, 65535)), or with ceil when up. A NaN quotient (only from a span
// beyond the float range) gives 0.
public uint ptExactQuantise(uint q, bool up) {
  if (ptExactIsNaN(q) || (q >> 31u) != 0u || q == 0u) return 0u;
  const uint exponent = (q >> 23u) & 0xFFu;
  if (exponent >= 127u + 16u) return 65535u;
  if (exponent < 127u) {
    if (up) return 1u;
    return 0u;
  }
  const uint significand = (q & 0x7FFFFFu) | 0x800000u;
  const uint shift = 150u - exponent;
  uint value = significand >> shift;
  if (up && (significand & ((1u << shift) - 1u)) != 0u) value += 1u;
  return min(value, 65535u);
}

// ptExactQuantise(ptExactDivide(n, d), up). For a normal divisor below 2^126 Vulkan's division
// is within 2.5 ULP of the exact quotient (and at most 3 ULP), and the correctly rounded one is
// within half an ULP of that; 8 steps span more than 4 ULP even across a binade. So when every
// float within 8 steps of the device's quotient falls in one bin, that is the bin; a quotient
// at a bin edge takes the exact division.
public uint ptExactQuantiseQuotient(uint n, uint d, bool up) {
  if (ptExactNormal(n) && ptExactNormal(d) && ptExactExponent(d) < 127 + 126) {
    const uint q = asuint(asfloat(n) / asfloat(d));
    if (ptExactNormal(q) && ptExactNormal(q - 8u) && ptExactNormal(q + 8u)) {
      const uint below = ptExactQuantise(q - 8u, up), above = ptExactQuantise(q + 8u, up);
      if (below == above) return below;
    }
  }
  return ptExactQuantise(ptExactDivide(n, d), up);
}
