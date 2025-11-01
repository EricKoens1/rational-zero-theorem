#!/usr/bin/env python3
"""
Rational Zero Theorem Calculator
Author: EricKoens1
Description: Parses polynomial strings, finds all possible rational zeros using
             the Rational Zero Theorem (p/q method). Handles decimal coefficients
             by converting to integers. Prepares for synthetic division.
"""

import re  # Regular expressions for parsing
from fractions import Fraction  # For exact rational arithmetic and display
from math import gcd  # Greatest common divisor for simplification
from functools import reduce  # For finding LCM


def parse_polynomial(poly_string):
    """
    Parses a polynomial string into a list of coefficients in descending order.

    Handles:
    - Spaces or no spaces: "x^2 + 3x - 5" or "x^2+3x-5"
    - Implicit coefficients: "x^2" means "1x^2", "-x" means "-1x"
    - Implicit exponents: "3x" means "3x^1"
    - Missing terms: fills with 0 coefficients
    - Decimal coefficients: "2.5x^2 + 3x - 1"

    Args:
        poly_string (str): The polynomial as a string

    Returns:
        tuple: (coefficients_list, degree, has_decimals)
               coefficients_list: [a_n, a_(n-1), ..., a_1, a_0]
               degree: highest power
               has_decimals: True if any decimal coefficients found

    Example:
        Input: "x^3 + 4x - 5"
        Output: ([1.0, 0.0, 4.0, -5.0], 3, False)
                 Represents: 1x^3 + 0x^2 + 4x^1 + (-5)x^0
    """
    # Remove all spaces for easier parsing
    poly_string = poly_string.replace(" ", "")

    # Remove "f(x)=" or "y=" if present
    poly_string = re.sub(r'^[fy]\(x\)=', '', poly_string, flags=re.IGNORECASE)
    poly_string = re.sub(r'^y=', '', poly_string, flags=re.IGNORECASE)

    # Dictionary to store {exponent: coefficient}
    terms = {}
    has_decimals = False

    # Regular expression to match polynomial terms
    # Matches: optional sign, coefficient (optional), x (optional), exponent (optional)
    # Examples: "+3x^2", "-x^3", "5x", "-7", "2.5x^4"
    pattern = r'([+-]?)(\d+\.?\d*)?(\*?)([xX])?(\^)?(\d+)?'

    # Find all terms in the polynomial
    matches = re.findall(pattern, poly_string)

    for match in matches:
        sign, coef, times, x, caret, exp = match

        # Skip empty matches
        if not any(match):
            continue

        # Determine the sign (default positive)
        sign_value = -1 if sign == '-' else 1

        # Determine coefficient
        if coef:
            coef_value = float(coef) * sign_value
            # Check if it's a decimal
            if '.' in coef:
                has_decimals = True
        elif x:  # If there's an x but no coefficient, it's 1 (or -1)
            coef_value = 1.0 * sign_value
        else:
            # No coefficient and no x means this might be empty, skip
            if not coef and not x:
                continue
            coef_value = 1.0 * sign_value

        # Determine exponent
        if x:  # There's an x variable
            if exp:  # Explicit exponent like x^3
                exp_value = int(exp)
            else:  # Just x, means x^1
                exp_value = 1
        else:  # No x, it's a constant term (x^0)
            if coef:  # Only if there's actually a coefficient
                exp_value = 0
            else:
                continue  # Skip if nothing meaningful

        # Add or update the term
        # If the exponent already exists, add coefficients (though this shouldn't happen in valid input)
        if exp_value in terms:
            terms[exp_value] += coef_value
        else:
            terms[exp_value] = coef_value

    # Find the highest degree
    if not terms:
        raise ValueError("No valid polynomial terms found")

    degree = max(terms.keys())

    # Create coefficient list from highest degree to 0
    # Fill missing degrees with 0
    coefficients = []
    for i in range(degree, -1, -1):
        coefficients.append(terms.get(i, 0.0))

    return coefficients, degree, has_decimals


def lcm(a, b):
    """
    Calculate Least Common Multiple of two numbers.

    Args:
        a, b: Two integers

    Returns:
        int: LCM of a and b
    """
    return abs(a * b) // gcd(a, b)


def find_decimal_multiplier(coefficients):
    """
    Finds the multiplier needed to convert all decimal coefficients to integers.

    Strategy:
    - Find how many decimal places each coefficient has
    - Calculate the power of 10 needed to clear all decimals

    Args:
        coefficients (list): List of float coefficients

    Returns:
        int: The multiplier (power of 10) to make all coefficients integers

    Example:
        [2.5, 3.0, 1.25] -> Need to multiply by 100 (max 2 decimal places)
        Result: [250, 300, 125]
    """
    max_decimals = 0

    for coef in coefficients:
        # Convert to string to count decimal places
        coef_str = str(abs(coef))

        if '.' in coef_str:
            # Count digits after decimal point
            decimal_part = coef_str.split('.')[1]
            # Remove trailing zeros for counting
            decimal_part = decimal_part.rstrip('0')
            decimals = len(decimal_part)
            max_decimals = max(max_decimals, decimals)

    # Return 10^max_decimals
    return 10 ** max_decimals


def convert_to_integers(coefficients):
    """
    Converts decimal coefficients to integers by finding appropriate multiplier.

    Args:
        coefficients (list): List of float coefficients

    Returns:
        tuple: (integer_coefficients, multiplier)
               integer_coefficients: List of integer coefficients
               multiplier: What we multiplied by
    """
    multiplier = find_decimal_multiplier(coefficients)

    # Multiply all coefficients
    integer_coeffs = [int(round(coef * multiplier)) for coef in coefficients]

    # Simplify by dividing by GCD if possible
    # This reduces unnecessarily large numbers
    common_gcd = reduce(gcd, [abs(c) for c in integer_coeffs if c != 0])

    if common_gcd > 1:
        integer_coeffs = [c // common_gcd for c in integer_coeffs]
        # Adjust multiplier to reflect simplification
        multiplier = multiplier // common_gcd

    return integer_coeffs, multiplier


def display_polynomial(coefficients, show_zeros=False):
    """
    Converts coefficient list back to readable polynomial string.

    Args:
        coefficients (list): List of coefficients in descending order
        show_zeros (bool): Whether to show terms with 0 coefficient

    Returns:
        str: Formatted polynomial string

    Example:
        [1, 0, -3, 5] -> "x^3 - 3x + 5"
        [1, 0, -3, 5] with show_zeros=True -> "x^3 + 0x^2 - 3x + 5"
    """
    degree = len(coefficients) - 1
    terms = []

    for i, coef in enumerate(coefficients):
        current_degree = degree - i

        # Skip zero coefficients unless show_zeros is True
        if coef == 0 and not show_zeros:
            continue

        # Build the term string
        term = ""

        # Handle the sign and coefficient
        if i == 0:  # First term
            if coef < 0:
                if current_degree > 0 and abs(coef) == 1:
                    term = "-"
                else:
                    term = str(int(coef)) if coef == int(coef) else str(coef)
            else:
                if current_degree > 0 and coef == 1:
                    term = ""
                else:
                    term = str(int(coef)) if coef == int(coef) else str(coef)
        else:  # Subsequent terms
            if coef < 0:
                if current_degree > 0 and abs(coef) == 1:
                    term = " - "
                else:
                    abs_coef = abs(coef)
                    term = f" - {int(abs_coef) if abs_coef == int(abs_coef) else abs_coef}"
            else:
                if current_degree > 0 and coef == 1:
                    term = " + "
                else:
                    term = f" + {int(coef) if coef == int(coef) else coef}"

        # Add the variable part
        if current_degree > 1:
            term += f"x^{current_degree}"
        elif current_degree == 1:
            term += "x"
        # If degree is 0, no variable (just the coefficient)

        terms.append(term)

    return "".join(terms) if terms else "0"


def find_factors(n):
    """
    Finds all positive factors of a number.
    (Reused logic from factor-finder project)

    Args:
        n (int): The number to find factors for

    Returns:
        list: Sorted list of positive factors
    """
    if n == 0:
        return [1]  # Special case: treat 0 as having factor 1

    n = abs(n)  # Work with positive value
    factors = []

    # Only check up to square root
    import math
    limit = int(math.sqrt(n)) + 1

    for i in range(1, limit):
        if n % i == 0:
            factors.append(i)
            paired_factor = n // i
            if paired_factor != i:
                factors.append(paired_factor)

    factors.sort()
    return factors


def calculate_rational_zeros(leading_coef, constant_term):
    """
    Calculates all possible rational zeros using the Rational Zero Theorem.

    Formula: p/q where p = factors of constant term, q = factors of leading coefficient

    Args:
        leading_coef (int): Leading coefficient (a_n)
        constant_term (int): Constant term (a_0)

    Returns:
        list: Sorted list of unique possible rational zeros as Fraction objects
    """
    # Get factors of constant term (p values)
    p_factors = find_factors(abs(constant_term))

    # Get factors of leading coefficient (q values)
    q_factors = find_factors(abs(leading_coef))

    # Calculate all p/q combinations
    possible_zeros = set()  # Use set to avoid duplicates

    for p in p_factors:
        for q in q_factors:
            # Add both positive and negative versions
            possible_zeros.add(Fraction(p, q))
            possible_zeros.add(Fraction(-p, q))

    # Convert to sorted list
    zeros_list = sorted(list(possible_zeros))

    return zeros_list


def display_theorem():
    """
    Displays the Rational Zero Theorem explanation.
    """
    print("\n" + "=" * 70)
    print("RATIONAL ZERO THEOREM")
    print("=" * 70)
    print("""
For a polynomial with integer coefficients:

    f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀

If p/q is a rational zero (where p and q have no common factors), then:

    • p is a factor of the constant term (a₀)
    • q is a factor of the leading coefficient (aₙ)

This theorem gives us ALL POSSIBLE rational zeros to test.
Not all of these will actually be zeros - they're candidates to check!
""")
    print("=" * 70 + "\n")


def main():
    """
    Main function to run the Rational Zero Theorem calculator.
    """
    print("\n" + "=" * 70)
    print("RATIONAL ZERO THEOREM CALCULATOR")
    print("=" * 70)
    print("\nEnter a polynomial function.")
    print("Examples:")
    print("  • x^3 - 15x^2 - 5x + 10")
    print("  • 3x^4 + 19x^3 + 20x^2 - 15x - 6")
    print("  • 2.5x^2 + 3x - 1  (decimals will be converted)")
    print()

    # Get polynomial input
    poly_input = input("Enter polynomial: ").strip()

    if not poly_input:
        print("Error: No input provided.")
        return

    try:
        # Parse the polynomial
        print("\n" + "-" * 70)
        print("STEP 1: PARSING POLYNOMIAL")
        print("-" * 70)

        coefficients, degree, has_decimals = parse_polynomial(poly_input)

        print(f"\nOriginal input: {poly_input}")
        print(f"Parsed as: {display_polynomial(coefficients, show_zeros=True)}")
        print(f"Degree: {degree}")
        print(f"Coefficients (descending order): {[int(c) if c == int(c) else c for c in coefficients]}")

        # Convert decimals to integers if necessary
        if has_decimals:
            print("\n" + "-" * 70)
            print("STEP 2: CONVERTING DECIMALS TO INTEGERS")
            print("-" * 70)
            print("\nRational Zero Theorem requires integer coefficients.")
            print("Converting by multiplying through...")

            coefficients, multiplier = convert_to_integers(coefficients)

            print(f"\nMultiplied all terms by: {multiplier}")
            print(f"Integer form: {display_polynomial(coefficients, show_zeros=True)}")
            print(f"Integer coefficients: {coefficients}")
            print("\nNote: This doesn't change the zeros of the function!")

        # Extract leading coefficient and constant term
        leading_coef = int(coefficients[0])
        constant_term = int(coefficients[-1])

        # Validate
        if leading_coef == 0:
            print("\nError: Leading coefficient cannot be zero!")
            return

        if constant_term == 0:
            print("\nNote: Constant term is 0, so x = 0 is automatically a zero.")
            print("For non-zero rational zeros, we can factor out x and analyze the remaining polynomial.")

        # Display the theorem
        display_theorem()

        # Show the calculation steps
        print("-" * 70)
        print("STEP 3: FINDING POSSIBLE RATIONAL ZEROS")
        print("-" * 70)

        print(f"\nLeading coefficient (aₙ): {leading_coef}")
        print(f"Constant term (a₀): {constant_term}")

        # Find factors
        p_factors = find_factors(abs(constant_term))
        q_factors = find_factors(abs(leading_coef))

        print(f"\nFactors of constant term ({constant_term}): ±{p_factors}")
        print(f"Factors of leading coefficient ({leading_coef}): ±{q_factors}")

        # Calculate rational zeros
        possible_zeros = calculate_rational_zeros(leading_coef, constant_term)

        print(f"\nPossible rational zeros (p/q):")
        print("-" * 70)

        # Display as fractions
        zeros_display = [str(z) for z in possible_zeros]
        print(", ".join(zeros_display))

        print(f"\nTotal possible rational zeros: {len(possible_zeros)}")

        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("""
To find which of these are ACTUAL zeros:
1. Test each candidate using synthetic division or substitution
2. If f(candidate) = 0, then it's a real zero!
3. Use synthetic division to factor and find remaining zeros
""")

    except ValueError as e:
        print(f"\nError parsing polynomial: {e}")
        print("Please check your input format.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
