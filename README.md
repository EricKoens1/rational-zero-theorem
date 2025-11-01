# Rational Zero Theorem Calculator

**Author:** EricKoens1
**Purpose:** Educational tool for finding possible rational zeros of polynomials - homework helper!

## 📋 Project Description

The Rational Zero Theorem Calculator is a multi-language implementation of an educational mathematics tool that finds all possible rational zeros of a polynomial function. This project helps students understand and apply the Rational Zero Theorem, preparing for synthetic division and complete polynomial solving.

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
- ✅ **Fraction Output:** Displays results as simplified fractions (e.g., "2/3" not "0.666...")
- ✅ **Educational:** Shows theorem explanation and step-by-step calculations
- ✅ **Synthetic Division Ready:** Stores coefficients in proper order for next steps
- ✅ **Extensively Commented:** Every function explained for learning

## 🚀 Quick Start

### Python Version

```bash
cd python
python3 rational_zeros.py
```

**Enter a polynomial like:**
- `x^3 - 15x^2 - 5x + 10`
- `3x^4 + 19x^3 + 20x^2 - 15x - 6`
- `2.5x^2 + 3x - 1` (decimals automatically converted)

## 📖 Examples

### Example 1: Basic Polynomial

**Input:**
```
x^3 - 15x^2 - 5x + 10
```

**Output:**
```
Parsed as: x^3 - 15x^2 - 5x + 10
Degree: 3
Coefficients: [1, -15, -5, 10]

Leading coefficient: 1
Constant term: 10

Factors of 10: ±[1, 2, 5, 10]
Factors of 1: ±[1]

Possible rational zeros:
-10, -5, -2, -1, 1, 2, 5, 10
```

### Example 2: Polynomial with Fractions

**Input:**
```
3x^4 + 19x^3 + 20x^2 - 15x - 6
```

**Output:**
```
Parsed as: 3x^4 + 19x^3 + 20x^2 - 15x - 6
Coefficients: [3, 19, 20, -15, -6]

Leading coefficient: 3
Constant term: -6

Factors of -6: ±[1, 2, 3, 6]
Factors of 3: ±[1, 3]

Possible rational zeros:
-6, -3, -2, -1, -2/3, -1/3, 1/3, 2/3, 1, 2, 3, 6
```

Note the fractions: **-2/3, -1/3, 1/3, 2/3** are displayed clearly!

### Example 3: Decimal Coefficients

**Input:**
```
2.5x^2 + 3x - 1
```

**Output:**
```
Original: 2.5x^2 + 3x - 1
Converting to integers (multiplying by 2):
Integer form: 5x^2 + 6x - 2

Leading coefficient: 5
Constant term: -2

Possible rational zeros:
-2, -1, -2/5, -1/5, 1/5, 2/5, 1, 2
```

The zeros are the same! The conversion is mathematically valid.

### Example 4: Missing Terms

**Input:**
```
x^5 + 4x^4 + x^3 + 10x + 100
```

**Output:**
```
Parsed as: x^5 + 4x^4 + x^3 + 0x^2 + 10x + 100
Coefficients: [1, 4, 1, 0, 10, 100]
```

Notice the **0x^2** term is added! This is crucial for synthetic division later.

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

### Step 3: Find Factors

Using the optimized √n algorithm (from factor-finder project):
- Find all factors of the constant term
- Find all factors of the leading coefficient

### Step 4: Calculate p/q Combinations

For each factor p of constant term and q of leading coefficient:
- Create fraction p/q
- Add both positive and negative versions
- Use Python's `Fraction` class for automatic simplification
- Remove duplicates (e.g., 6/3 = 2)

### Step 5: Display Results

Show all possible rational zeros sorted from smallest to largest, as fractions.

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
- Factor pairs and factor finding
- Polynomial structure and terms
- Converting decimals to integers while preserving zeros
- Why the theorem gives candidates (not guaranteed zeros)

**Programming Concepts (Python):**
- Regular expressions for parsing
- String manipulation and pattern matching
- Fraction arithmetic with Python's `fractions` module
- Set data structures (for removing duplicates)
- List comprehensions
- Error handling and validation

### Preparing for Synthetic Division

The coefficient array is stored in descending order:

```python
Input: "3x^3 + x^2 + 10x + 2"
Stored as: [3, 1, 10, 2]
```

This is **exactly** the format needed for synthetic division! You can directly use these coefficients to test each possible zero.

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
✅ Polynomials with fractional results
✅ Decimal coefficient conversion
✅ Missing terms (fills with 0)
✅ No spaces in input
✅ Negative leading coefficient
✅ Implicit coefficients (x^2, -x)

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

### Phase 2: Synthetic Division
- [ ] Implement synthetic division algorithm
- [ ] Automatically test which possible zeros are actual zeros
- [ ] Factor the polynomial completely
- [ ] Find all real and complex zeros

### Phase 3: Enhanced Features
- [ ] Graphing capabilities
- [ ] Web interface (JavaScript version)
- [ ] Step-by-step synthetic division walkthrough
- [ ] Export results to LaTeX format

### Phase 4: Advanced Math
- [ ] Handle complex coefficients
- [ ] Descartes' Rule of Signs
- [ ] Upper/Lower bound testing
- [ ] Numerical approximation for irrational zeros

## 🎯 Use Cases

- **Homework Helper:** Find candidates before synthetic division
- **Test Preparation:** Practice applying the theorem
- **Teaching Tool:** Demonstrate the theorem step-by-step
- **Verification:** Check your manual calculations
- **Learning Programming:** Compare implementations across languages

## 💡 Tips for Using This Tool

1. **After getting possible zeros, test them!**
   - Use synthetic division
   - Or substitute into the original polynomial: f(candidate) = 0?

2. **Not all possible zeros are actual zeros**
   - The theorem gives you candidates to test
   - Some (or all) might not actually be zeros

3. **Decimal handling is mathematically sound**
   - The converted polynomial has the same zeros
   - You can trust the results!

4. **Use the coefficient array for synthetic division**
   - The array is ready to use directly
   - Order is correct (descending degree)

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
