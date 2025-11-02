# Rational Zero Theorem Calculator

**Author:** EricKoens1
**Purpose:** Educational tool for finding possible rational zeros of polynomials - homework helper!

## 📋 Project Description

The Rational Zero Theorem Calculator is a powerful educational mathematics tool that **completely solves polynomial equations**, finding ALL zeros (rational, irrational, and complex) with exact radical forms. Starting from the Rational Zero Theorem, it uses synthetic division and the quadratic formula to factor polynomials completely and display all zeros in simplified form—perfect for homework assignments!

**What makes it special:**
- 🎯 **Complete automation:** Enter polynomial → Get all zeros instantly
- ✏️ **Exact answers:** All zeros displayed with radicals (√) and i, fully simplified
- 📚 **Educational modes:** Step-by-step walkthrough OR quick answer
- ✅ **Homework ready:** Copy exact forms directly into assignments

### What is the Rational Zero Theorem?

For a polynomial with integer coefficients:

```
f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀
```

If **p/q** is a rational zero (in lowest terms), then:
- **p** is a factor of the constant term (a₀)
- **q** is a factor of the leading coefficient (aₙ)

This gives us a finite list of **candidates** to test as potential zeros!

## 🎯 Features

- ✅ **Flexible Input:** Parses polynomial strings with or without spaces
- ✅ **Handles Decimals:** Automatically converts to integers (mathematically correct)
- ✅ **Fills Missing Terms:** Adds 0 coefficients for missing degrees
- ✅ **Mode Selection:** Choose between step-by-step (educational) or quick answer mode
- ✅ **Complete Factorization:** Automatically finds ALL zeros (rational, irrational, and complex)
- ✅ **Exact Radical Forms:** Displays zeros using exact radicals and i (perfect for homework!)
- ✅ **Full Simplification:** Simplifies all radicals (√24 = 2√6) and applies GCD to all components
- ✅ **Synthetic Division:** Uses optimized algorithm to test possible zeros and factor recursively
- ✅ **Quadratic Formula:** Handles three cases - rational, irrational real, and complex zeros
- ✅ **Comprehensive Output:** Shows degree, all possible zeros, factored form, and complete zero list
- ✅ **Fraction Output:** Displays results as simplified fractions (e.g., "2/3" not "0.666...")
- ✅ **Educational:** Shows theorem explanation and step-by-step calculations
- ✅ **Extensively Commented:** Every function explained for learning

## 🚀 Quick Start

### Python Version

```bash
cd python
python3 rational_zeros.py
```

**Choose your mode:**
1. **Step-by-step mode**: See the Rational Zero Theorem explanation, factor finding, synthetic division process, and all intermediate steps
2. **Quick answer mode**: Get straight to the complete factorization and all zeros

**Enter a polynomial like:**
- `x^3 - 15x^2 - 5x + 10`
- `3x^4 + 19x^3 + 20x^2 - 15x - 6`
- `2.5x^2 + 3x - 1` (decimals automatically converted)
- `11x^3 + 126x^2 + 56x + 11` (includes complex zeros)
- `x^2 - 2x - 1` (includes irrational zeros)

## 📖 Examples

### Example 1: Complete Factorization with Rational Zeros

**Input:**
```
x^3 - 6x^2 + 11x - 6
```

**Output (Quick Mode):**
```
========================================
COMPLETE FACTORIZATION
========================================

Original polynomial: x^3 - 6x^2 + 11x - 6
Degree: 3

Possible rational zeros (from Rational Zero Theorem):
  -6, -3, -2, -1, 1, 2, 3, 6
  Total candidates: 8

Factored form: (x - 1)(x - 2)(x - 3)

All zeros found:
  x = 1
  x = 2
  x = 3
```

### Example 2: Polynomial with Complex Zeros (Exact Radical Form)

**Input:**
```
11x^3 + 126x^2 + 56x + 11
```

**Output (Quick Mode):**
```
========================================
COMPLETE FACTORIZATION
========================================

Original polynomial: 11x^3 + 126x^2 + 56x + 11
Degree: 3

Possible rational zeros (from Rational Zero Theorem):
  -11, -1, -1/11, 1/11, 1, 11
  Total candidates: 6

Factored form: (x + 11) × (complex quadratic factor)

All zeros found:
  x = -11
  x = (-5 + i√19) / 22 (complex)   ≈ -0.0277 + 0.1978i
  x = (-5 - i√19) / 22 (complex)   ≈ -0.0277 - 0.1978i
```

Note: Complex zeros are displayed in **exact radical form** using i, perfect for homework! The radical √19 cannot be simplified further, and all components are reduced by their GCD.

### Example 3: Irrational Zeros with Simplified Radicals

**Input:**
```
x^2 - 2x - 1
```

**Output (Quick Mode):**
```
========================================
COMPLETE FACTORIZATION
========================================

Original polynomial: x^2 - 2x - 1
Degree: 2

Possible rational zeros (from Rational Zero Theorem):
  -1, 1
  Total candidates: 2

Factored form: (irrational quadratic factor)

All zeros found:
  x = 1 + √2 (irrational)   ≈ 2.4142
  x = 1 - √2 (irrational)   ≈ -0.4142
```

Note: The radical √2 is already in simplest form. If the discriminant were 24, it would be simplified to 2√6 automatically!

### Example 4: Radical Simplification Showcase

**Input:**
```
7x^2 + 2x - 1
```

**Output (Quick Mode):**
```
========================================
COMPLETE FACTORIZATION
========================================

Original polynomial: 7x^2 + 2x - 1
Degree: 2

Possible rational zeros (from Rational Zero Theorem):
  -1, 1, -1/7, 1/7
  Total candidates: 4

Factored form: (irrational quadratic factor)

All zeros found:
  x = (-1 + 2√2) / 7 (irrational)   ≈ 0.2612
  x = (-1 - 2√2) / 7 (irrational)   ≈ -0.5469
```

**Simplification Process Shown:**
- Discriminant = 4 + 28 = 32 = 16 × 2
- √32 extracts perfect square: √(16 × 2) = 4√2
- Apply GCD to (-2, 4, 14) = 2
- Result: (-1 + 2√2) / 7 ✓ Fully simplified!

## 🧮 How It Works

### Step 1: Parse the Polynomial

The parser handles:
- **Spaces or no spaces:** `x^2+3x-5` or `x^2 + 3x - 5`
- **Implicit coefficients:** `x^2` means `1x^2`, `-x` means `-1x`
- **Implicit exponents:** `3x` means `3x^1`
- **Constants:** `5` means `5x^0`

### Step 2: Convert Decimals to Integers

If decimals are found:
1. Find the least common multiplier to clear all decimals
2. Multiply all coefficients by this multiplier
3. Simplify by dividing by GCD if possible

**Why this works:** If f(x) = 0, then k·f(x) = 0 for any constant k. The zeros don't change!

### Step 3: Find Factors (Rational Zero Theorem)

Using the optimized √n algorithm (from factor-finder project):
- Find all factors of the constant term
- Find all factors of the leading coefficient
- Create all p/q combinations (with simplification)
- Sort and display possible rational zeros

### Step 4: Test Zeros with Synthetic Division

**Recursive Algorithm:**
1. Test each possible zero using synthetic division
2. When a zero is found, collect it and the quotient polynomial
3. Recursively factor the quotient using the same possible_zeros list
4. Detect repeated roots by testing the last_zero first
5. Continue until quotient is degree 2 or lower

**Optimization:** Reusing the original possible_zeros list throughout saves computation!

### Step 5: Handle Quadratic Remainders (Quadratic Formula)

When the quotient reaches degree 2, apply the quadratic formula with three cases:

**Case 1: Discriminant = 0 (Rational - repeated root)**
```
Returns: Two identical rational zeros
```

**Case 2: Discriminant > 0 (Irrational - two real zeros)**
```
Process:
1. Simplify radical: √32 → 4√2, √24 → 2√6
2. Apply GCD to (numerator_const, radical_coef, denominator)
3. Format: (-1 + 2√2) / 7
```

**Case 3: Discriminant < 0 (Complex - conjugate pair)**
```
Process:
1. Calculate imaginary_squared = -discriminant
2. Simplify radical: √24 → 2√6
3. Apply GCD to (real_part, imaginary_coef, denominator)
4. Format: (-1 + i√6) / 7 and (-1 - i√6) / 7
```

### Step 6: Display Complete Results

Show comprehensive output:
- Original polynomial and degree
- All possible rational zeros (candidates)
- Complete factored form
- All zeros found (rational, irrational, complex) with exact radical forms
- Decimal approximations for irrational/complex zeros

## 📂 Project Structure

```
rational-zero-theorem/
├── README.md           # This file
├── python/
│   └── rational_zeros.py   # Python implementation
├── c/                  # Coming soon!
│   └── (future)
└── javascript/         # Coming soon!
    └── (future)
```

## 🎓 Educational Value

### What You'll Learn

**Mathematical Concepts:**
- Rational Zero Theorem application
- Synthetic division algorithm
- Recursive polynomial factorization
- Quadratic formula with three cases (rational, irrational, complex)
- Radical simplification (extracting perfect square factors)
- GCD (Greatest Common Divisor) for fraction reduction
- Complex numbers in standard form (a + bi)
- Factor pairs and optimized factor finding
- Polynomial structure and degree
- Converting decimals to integers while preserving zeros
- Detecting and handling repeated roots (multiplicity)

**Programming Concepts (Python):**
- Regular expressions for parsing complex patterns
- Recursive algorithms with state preservation
- String manipulation and pattern matching
- Fraction arithmetic with Python's `fractions` module
- Complex number formatting and simplification
- Set data structures (for removing duplicates)
- List comprehensions and functional programming
- Error handling and input validation
- Algorithm optimization (reusing computed values)
- Math functions: `sqrt()`, `gcd()`, `isqrt()`

### Complete Polynomial Solving

The program doesn't just find possible zeros—it **solves the polynomial completely**:

1. **Finds all rational zeros** using Rational Zero Theorem + Synthetic Division
2. **Handles irrational zeros** using Quadratic Formula with exact radical forms
3. **Handles complex zeros** with simplified a + bi format
4. **Displays factored form** showing all linear and irreducible quadratic factors
5. **Shows approximations** for irrational and complex zeros

Perfect for homework assignments that require exact answers!

## 🔧 Requirements

### Python
- Python 3.6+ (uses f-strings)
- No external dependencies (only standard library)
- Built-in modules used:
  - `re` - Regular expressions
  - `fractions` - Exact rational arithmetic
  - `math` - GCD function
  - `functools` - Reduce for LCM

## 🧪 Testing

All test cases verified:

✅ Basic polynomials with integer coefficients
✅ Complete factorization with all rational zeros
✅ Polynomials with fractional rational zeros (p/q form)
✅ Polynomials with irrational real zeros (exact radical form)
✅ Polynomials with complex zeros (a + bi with radicals)
✅ Radical simplification (√24 → 2√6, √32 → 4√2)
✅ GCD simplification for all zero types
✅ Repeated roots (multiplicity detection)
✅ Decimal coefficient conversion
✅ Missing terms (automatic 0 coefficient insertion)
✅ No spaces in input
✅ Negative leading coefficient
✅ Implicit coefficients (x^2, -x)
✅ Mode selection (step-by-step vs quick)
✅ Edge cases: i, -i, √2, etc.

**Test Examples from Actual Homework:**
- `11x^3 + 126x^2 + 56x + 11` → Complex zeros: `(-5 ± i√19) / 22` ✓
- `7x^2 + 2x - 1` → Irrational: `(-1 ± 2√2) / 7` ✓
- `x^3 - 6x^2 + 11x - 6` → Rational: `1, 2, 3` ✓

## 📐 Algorithm Complexity

- **Parsing:** O(n) where n is the length of input string
- **Factor finding:** O(√p + √q) where p is constant term, q is leading coefficient
- **p/q calculation:** O(m × n) where m = number of p factors, n = number of q factors
- **Sorting:** O(k log k) where k = number of unique rational zeros

**Overall:** Very efficient even for large coefficients!

## 🔮 Future Enhancements

### Phase 1: More Languages
- [ ] C implementation
- [ ] JavaScript implementation
- [ ] Go implementation (concurrency for testing zeros)

### Phase 2: Synthetic Division ✅ COMPLETED!
- [x] Implement synthetic division algorithm
- [x] Automatically test which possible zeros are actual zeros
- [x] Factor the polynomial completely
- [x] Find all real and complex zeros
- [x] Handle repeated roots (multiplicity)
- [x] Display exact radical forms for irrational/complex zeros
- [x] Implement full simplification (radicals + GCD)

### Phase 3: Enhanced Features
- [x] Mode selection (step-by-step vs quick answer)
- [x] Comprehensive output display
- [ ] Graphing capabilities
- [ ] Web interface (JavaScript version)
- [ ] Export results to LaTeX format

### Phase 4: Advanced Math
- [ ] Handle complex coefficients (input)
- [ ] Descartes' Rule of Signs
- [ ] Upper/Lower bound testing
- [ ] Numerical root refinement (Newton's method)
- [ ] Polynomial division for arbitrary divisors

## 🎯 Use Cases

- **Homework Helper:** Get exact answers in simplified radical form - ready to submit!
- **Complete Solutions:** Finds ALL zeros automatically (no manual synthetic division needed)
- **Test Preparation:** Learn the complete polynomial solving process
- **Teaching Tool:** Step-by-step mode shows the entire Rational Zero Theorem workflow
- **Answer Verification:** Check your manual calculations against exact solutions
- **Learning Programming:** Study recursive algorithms and mathematical formatting

## 💡 Tips for Using This Tool

1. **Choose the right mode:**
   - **Step-by-step:** Great for learning and understanding the process
   - **Quick answer:** Perfect when you just need the solution for homework

2. **Exact answers for homework:**
   - All zeros are displayed in exact form (radicals and i)
   - Fully simplified automatically (radicals extracted, GCD applied)
   - Copy the exact form directly into your assignment!

3. **Understanding the output:**
   - **Rational zeros:** Simple fractions or integers (e.g., `x = 2`, `x = -1/3`)
   - **Irrational zeros:** Use √ notation (e.g., `x = 1 + √2`, `x = (-1 + 2√2) / 7`)
   - **Complex zeros:** Use i notation (e.g., `x = i`, `x = (-5 + i√19) / 22`)
   - Approximations are shown for reference, but use the exact form!

4. **Decimal handling is mathematically sound:**
   - The converted polynomial has the same zeros
   - You can trust the results!

5. **Factored form interpretation:**
   - Linear factors: `(x - a)` where `a` is a rational zero
   - Irrational/complex factors: Shown as "quadratic factor" (cannot be factored over rationals)

## 📚 Mathematical Background

### Why Does This Work?

If p/q is a rational root of:
```
aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀ = 0
```

Then substituting x = p/q and multiplying by qⁿ gives:
```
aₙpⁿ + aₙ₋₁pⁿ⁻¹q + ... + a₁pqⁿ⁻¹ + a₀qⁿ = 0
```

Rearranging:
```
aₙpⁿ = -q(aₙ₋₁pⁿ⁻¹ + ... + a₀qⁿ⁻¹)
```

This shows **p divides aₙpⁿ**, and since p and q share no common factors, **p must divide a₀**.

Similarly, **q must divide aₙ**.

## 🤝 Contributing

This is a personal learning project, but suggestions welcome!

## 📄 License

Educational purposes - free to use and modify for learning!

---

**Created:** November 2025
**Last Updated:** November 2025
**Repository:** https://github.com/EricKoens1/rational-zero-theorem

**Related Projects:**
- [Factor Finder](https://github.com/EricKoens1/factor-finder) - Multi-language factor finding (used in this project!)
