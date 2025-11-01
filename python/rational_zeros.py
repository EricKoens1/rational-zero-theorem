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


def synthetic_division(coefficients, candidate):
    """
    Performs synthetic division to test if a candidate is a zero.

    Process:
    1. Bring down the first coefficient
    2. Multiply by candidate, add to next coefficient
    3. Repeat for all coefficients
    4. Last value is the remainder (0 means candidate IS a zero!)

    Args:
        coefficients (list): Polynomial coefficients [aₙ, aₙ₋₁, ..., a₁, a₀]
        candidate (Fraction): The value to test (can be integer or fraction)

    Returns:
        tuple: (quotient_coefficients, remainder)
               - quotient_coefficients: Result after division (degree n-1)
               - remainder: If 0, candidate is a zero!

    Example:
        coefficients = [1, -5, 2, 8]
        candidate = 2
        Result: quotient = [1, -3, -4], remainder = 0
        Means: (x - 2)(x² - 3x - 4) = x³ - 5x² + 2x + 8
    """
    # Convert candidate to Fraction if not already
    if not isinstance(candidate, Fraction):
        candidate = Fraction(candidate)

    # Start with empty result row
    result = []
    current_value = Fraction(0)

    # Process each coefficient
    for i, coef in enumerate(coefficients):
        if i == 0:
            # First step: just bring down the first coefficient
            current_value = Fraction(coef)
        else:
            # Multiply previous result by candidate and add current coefficient
            current_value = current_value * candidate + Fraction(coef)

        result.append(current_value)

    # The last value is the remainder
    remainder = result[-1]

    # All values except the last are the quotient coefficients
    quotient = result[:-1]

    return quotient, remainder


def display_synthetic_division_interactive(coefficients, candidate):
    """
    Displays the synthetic division process step-by-step with user interaction.
    User presses Enter to advance through each step.

    Args:
        coefficients (list): Original polynomial coefficients
        candidate (Fraction): The tested value

    Returns:
        tuple: (quotient, remainder)
    """
    print("\n" + "━" * 70)
    print(f"Testing candidate: x = {candidate}")
    print("━" * 70)

    # Convert candidate to Fraction
    c = Fraction(candidate) if not isinstance(candidate, Fraction) else candidate

    # Determine column width
    all_numbers = [str(c)] + [str(Fraction(x)) for x in coefficients]
    col_width = max(len(str(n)) for n in all_numbers) + 2

    print("\nSynthetic Division - Step by Step")
    print("(Press Enter to continue through each step)\n")

    # Print the setup
    print(f"{str(c):>{col_width}} |", end="")
    for coef in coefficients:
        print(f"{str(Fraction(coef)):>{col_width}}", end="")
    print("\n" + " " * (col_width) + "|")

    input("Press Enter to begin...")

    # Step-by-step execution
    result = []
    multiply_row = []
    current_value = Fraction(0)

    for i, coef in enumerate(coefficients):
        if i == 0:
            # Step 1: Bring down first coefficient
            print(f"\n{'─' * 70}")
            print(f"STEP 1: Bring down the leading coefficient")
            print(f"{'─' * 70}")
            current_value = Fraction(coef)
            result.append(current_value)
            multiply_row.append(Fraction(0))

            print(f"\nBring down: {current_value}")
            print(f"\nCurrent result row: [{current_value}]")

            if i < len(coefficients) - 1:
                input("\nPress Enter to continue...")

        else:
            # Step 2+: Multiply and add
            step_number = i + 1
            print(f"\n{'─' * 70}")
            print(f"STEP {step_number}: Multiply and Add")
            print(f"{'─' * 70}")

            # Multiply previous result by candidate
            product = result[-1] * c
            multiply_row.append(product)

            print(f"\nMultiply: {result[-1]} × {c} = {product}")
            print(f"Add to next coefficient: {product} + {Fraction(coef)} = {product + Fraction(coef)}")

            # Add to current coefficient
            current_value = product + Fraction(coef)
            result.append(current_value)

            print(f"\nCurrent result row: {[str(x) for x in result]}")

            if i < len(coefficients) - 1:
                input("\nPress Enter to continue...")

    # Display final table
    print(f"\n{'═' * 70}")
    print("FINAL RESULT")
    print(f"{'═' * 70}\n")

    # Print complete table
    print(f"{str(c):>{col_width}} |", end="")
    for coef in coefficients:
        print(f"{str(Fraction(coef)):>{col_width}}", end="")
    print()

    print(f"{' ' * (col_width)} |", end="")
    for val in multiply_row:
        if val == 0:
            print(f"{' ' * col_width}", end="")
        else:
            print(f"{str(val):>{col_width}}", end="")
    print()

    print(f"{' ' * (col_width)} " + "─" * (col_width * len(coefficients) + 2))

    print(f"{' ' * (col_width + 1)}", end="")
    for val in result:
        print(f"{str(val):>{col_width}}", end="")
    print()

    # Extract quotient and remainder
    quotient = result[:-1]
    remainder = result[-1]

    print(f"\nQuotient coefficients: {[str(x) for x in quotient]}")
    print(f"Remainder: {remainder}")

    # Display result
    if remainder == 0:
        print(f"\n✓ x = {candidate} IS a zero!")
    else:
        print(f"\n✗ x = {candidate} is NOT a zero (remainder ≠ 0)")

    return quotient, remainder


def display_synthetic_division_quick(coefficients, candidate, quotient, remainder):
    """
    Displays the synthetic division result quickly without step-by-step.

    Args:
        coefficients (list): Original polynomial coefficients
        candidate (Fraction): The tested value
        quotient (list): Resulting quotient coefficients
        remainder (Fraction): The remainder from division
    """
    print("\n" + "━" * 70)
    print(f"Testing candidate: x = {candidate}")
    print("━" * 70)

    print("\nSynthetic Division:\n")

    # Convert candidate to Fraction for display
    c = Fraction(candidate) if not isinstance(candidate, Fraction) else candidate

    # Calculate the multiplication row (what we add at each step)
    multiply_row = [Fraction(0)]  # First entry is always 0 (nothing to multiply yet)
    current = Fraction(coefficients[0])  # Start with first coefficient

    for i in range(1, len(coefficients)):
        product = current * c
        multiply_row.append(product)
        current = current + Fraction(coefficients[i])

    # Determine column width based on longest number
    all_numbers = (
        [str(c)] +
        [str(Fraction(x)) for x in coefficients] +
        [str(x) for x in multiply_row] +
        [str(x) for x in quotient] +
        [str(remainder)]
    )
    col_width = max(len(str(n)) for n in all_numbers) + 2

    # Print candidate and first coefficient row
    print(f"{str(c):>{col_width}} |", end="")
    for coef in coefficients:
        print(f"{str(Fraction(coef)):>{col_width}}", end="")
    print()

    # Print multiplication row
    print(f"{' ' * (col_width)} |", end="")
    for val in multiply_row:
        if val == 0:
            print(f"{' ' * col_width}", end="")
        else:
            print(f"{str(val):>{col_width}}", end="")
    print()

    # Print separator line
    print(f"{' ' * (col_width)} " + "─" * (col_width * len(coefficients) + 2))

    # Print result row (quotient + remainder)
    print(f"{' ' * (col_width + 1)}", end="")
    for val in quotient:
        print(f"{str(val):>{col_width}}", end="")
    print(f"{str(remainder):>{col_width}}")

    # Print remainder info
    print(f"\nRemainder: {remainder}")

    # Display result
    if remainder == 0:
        print(f"\n✓ x = {candidate} IS a zero!")
    else:
        print(f"\n✗ x = {candidate} is NOT a zero (remainder ≠ 0)")


def solve_linear(coefficients):
    """
    Solves a linear equation ax + b = 0.

    Args:
        coefficients (list): [a, b] where equation is ax + b = 0

    Returns:
        Fraction: The solution x = -b/a
    """
    a = Fraction(coefficients[0])
    b = Fraction(coefficients[1])

    # ax + b = 0
    # ax = -b
    # x = -b/a
    return -b / a


def simplify_radical(n):
    """
    Simplifies √n by extracting perfect square factors.

    Args:
        n (int): The number under the radical

    Returns:
        tuple: (coefficient, remaining_radical)
               e.g., √24 = 2√6 returns (2, 6)
    """
    import math

    if n <= 0:
        return (1, n)

    # Find the largest perfect square factor
    coefficient = 1
    remaining = n

    # Check each potential factor up to sqrt(n)
    i = 2
    while i * i <= remaining:
        count = 0
        while remaining % (i * i) == 0:
            remaining //= (i * i)
            coefficient *= i
        i += 1

    return (coefficient, remaining)


def quadratic_formula(coefficients, verbose=True):
    """
    Solves a quadratic equation ax² + bx + c = 0 using the quadratic formula.

    Args:
        coefficients (list): [a, b, c] where equation is ax² + bx + c = 0
        verbose (bool): Whether to print detailed steps

    Returns:
        list: Solutions (may be real or complex)
    """
    a = Fraction(coefficients[0])
    b = Fraction(coefficients[1])
    c = Fraction(coefficients[2])

    # Calculate discriminant: b² - 4ac
    discriminant = b*b - 4*a*c

    if verbose:
        print(f"\nUsing Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a")
        print(f"  a = {a}, b = {b}, c = {c}")
        print(f"  Discriminant (b² - 4ac) = {discriminant}")

    if discriminant > 0:
        # Two real solutions
        import math
        sqrt_disc = math.sqrt(float(discriminant))

        # Try to express as exact fraction if possible
        x1 = (-b + Fraction(sqrt_disc).limit_denominator()) / (2*a)
        x2 = (-b - Fraction(sqrt_disc).limit_denominator()) / (2*a)

        if verbose:
            print(f"  Two real solutions (irrational)")
        return [f"({-b} + √{discriminant}) / {2*a}", f"({-b} - √{discriminant}) / {2*a}"], "real"

    elif discriminant == 0:
        # One real solution (repeated)
        x = -b / (2*a)
        if verbose:
            print(f"  One repeated real solution")
        return [x], "rational"

    else:
        # Two complex solutions: x = (-b ± i√(-discriminant)) / (2a)
        import math

        # For complex: discriminant < 0, so -discriminant > 0
        imag_part_squared = -discriminant

        if verbose:
            print(f"  Two complex solutions")

        # Simplify perfect squares under the radical
        sqrt_val = math.sqrt(float(imag_part_squared))
        is_perfect_square = (sqrt_val == int(sqrt_val))

        if is_perfect_square:
            # Simplify: √k = integer
            sqrt_simplified = int(sqrt_val)

            if b == 0:
                # No real part: x = ±i*k / (2a)
                denom = int(2*a)
                if denom == 1:
                    if sqrt_simplified == 1:
                        return ["i", "-i"], "complex"
                    else:
                        return [f"{sqrt_simplified}i", f"-{sqrt_simplified}i"], "complex"
                else:
                    if sqrt_simplified == 1:
                        if denom == 1:
                            return ["i", "-i"], "complex"
                        return [f"i / {denom}", f"-i / {denom}"], "complex"
                    else:
                        # Simplify fraction if possible
                        from math import gcd
                        g = gcd(sqrt_simplified, int(2*a))
                        num = sqrt_simplified // g
                        denom = int(2*a) // g

                        # Further simplify if numerator or denominator is 1
                        if num == 1 and denom == 1:
                            return ["i", "-i"], "complex"
                        elif denom == 1:
                            return [f"{num}i", f"-{num}i"], "complex"
                        elif num == 1:
                            return [f"i / {denom}", f"-i / {denom}"], "complex"
                        else:
                            return [f"{num}i / {denom}", f"-{num}i / {denom}"], "complex"
            else:
                # Has real part: (-b ± ki) / (2a) where k = sqrt_simplified
                from math import gcd

                # Find GCD of -b, sqrt_simplified, and 2a to simplify the entire expression
                g = gcd(gcd(abs(int(-b)), sqrt_simplified), int(2*a))

                real_part = int(-b) // g
                imag_part = sqrt_simplified // g
                denom = int(2*a) // g

                if denom == 1:
                    if imag_part == 1:
                        return [f"{real_part} + i", f"{real_part} - i"], "complex"
                    else:
                        return [f"{real_part} + {imag_part}i", f"{real_part} - {imag_part}i"], "complex"
                else:
                    if imag_part == 1:
                        return [f"({real_part} + i) / {denom}", f"({real_part} - i) / {denom}"], "complex"
                    else:
                        return [f"({real_part} + {imag_part}i) / {denom}", f"({real_part} - {imag_part}i) / {denom}"], "complex"
        else:
            # Not a perfect square - simplify the radical: √k = coef√remaining
            # e.g., √24 = 2√6
            radical_coef, radical_remaining = simplify_radical(int(imag_part_squared))

            if b == 0:
                # No real part: x = ±(radical_coef)i√k / (2a)
                from math import gcd
                # Simplify coefficient with denominator
                g = gcd(radical_coef, int(2*a))
                num = radical_coef // g
                denom = int(2*a) // g

                if denom == 1:
                    if num == 1:
                        if radical_remaining == 1:
                            return ["i", "-i"], "complex"
                        else:
                            return [f"i√{radical_remaining}", f"-i√{radical_remaining}"], "complex"
                    else:
                        if radical_remaining == 1:
                            return [f"{num}i", f"-{num}i"], "complex"
                        else:
                            return [f"{num}i√{radical_remaining}", f"-{num}i√{radical_remaining}"], "complex"
                else:
                    if num == 1:
                        if radical_remaining == 1:
                            return [f"i / {denom}", f"-i / {denom}"], "complex"
                        else:
                            return [f"i√{radical_remaining} / {denom}", f"-i√{radical_remaining} / {denom}"], "complex"
                    else:
                        if radical_remaining == 1:
                            return [f"{num}i / {denom}", f"-{num}i / {denom}"], "complex"
                        else:
                            return [f"{num}i√{radical_remaining} / {denom}", f"-{num}i√{radical_remaining} / {denom}"], "complex"
            else:
                # Has real part: x = (-b ± (radical_coef)i√k) / (2a)
                from math import gcd
                # Find GCD of all components
                g = gcd(gcd(abs(int(-b)), radical_coef), int(2*a))

                real_part = int(-b) // g
                imag_coef = radical_coef // g
                denom = int(2*a) // g

                if denom == 1:
                    if imag_coef == 1:
                        if radical_remaining == 1:
                            return [f"{real_part} + i", f"{real_part} - i"], "complex"
                        else:
                            return [f"{real_part} + i√{radical_remaining}", f"{real_part} - i√{radical_remaining}"], "complex"
                    else:
                        if radical_remaining == 1:
                            return [f"{real_part} + {imag_coef}i", f"{real_part} - {imag_coef}i"], "complex"
                        else:
                            return [f"{real_part} + {imag_coef}i√{radical_remaining}", f"{real_part} - {imag_coef}i√{radical_remaining}"], "complex"
                else:
                    if imag_coef == 1:
                        if radical_remaining == 1:
                            return [f"({real_part} + i) / {denom}", f"({real_part} - i) / {denom}"], "complex"
                        else:
                            return [f"({real_part} + i√{radical_remaining}) / {denom}", f"({real_part} - i√{radical_remaining}) / {denom}"], "complex"
                    else:
                        if radical_remaining == 1:
                            return [f"({real_part} + {imag_coef}i) / {denom}", f"({real_part} - {imag_coef}i) / {denom}"], "complex"
                        else:
                            return [f"({real_part} + {imag_coef}i√{radical_remaining}) / {denom}", f"({real_part} - {imag_coef}i√{radical_remaining}) / {denom}"], "complex"


def find_all_zeros_recursive(coefficients, possible_zeros, step_by_step=False, last_zero=None):
    """
    Recursively finds all rational zeros of a polynomial.

    Args:
        coefficients (list): Current polynomial coefficients
        possible_zeros (list): List of possible rational zeros (from ORIGINAL polynomial)
        step_by_step (bool): Whether to show step-by-step synthetic division
        last_zero (Fraction): The last zero found (to test first for multiplicity)

    Returns:
        list: All zeros found (as tuples of (zero, multiplicity))
    """
    degree = len(coefficients) - 1
    zeros_found = []

    if step_by_step:
        print(f"\n{'─' * 70}")
        print(f"Current polynomial degree: {degree}")
        print(f"Coefficients: {[str(c) if isinstance(c, Fraction) else c for c in coefficients]}")
        print(f"Polynomial: {display_polynomial(coefficients, show_zeros=False)}")
        print(f"{'─' * 70}")

    # Base cases
    if degree == 1:
        # Linear: solve directly
        zero = solve_linear(coefficients)
        if step_by_step:
            print(f"\n✓ Linear equation: {coefficients[0]}x + {coefficients[1]} = 0")
            print(f"  Solution: x = {zero}")
        return [(zero, 1)]

    elif degree == 2:
        # Quadratic: try rational zeros first, then quadratic formula
        if step_by_step:
            print(f"\nQuadratic polynomial detected. Testing rational zeros first...")

        # Reorder to test last_zero first if it exists
        test_order = possible_zeros.copy()
        if last_zero and last_zero in test_order:
            test_order.remove(last_zero)
            test_order.insert(0, last_zero)

        # Try to find rational zero
        for candidate in test_order:
            quotient, remainder = synthetic_division(coefficients, candidate)

            if remainder == 0:
                if step_by_step:
                    print(f"\n✓ Found rational zero: x = {candidate}")

                # Solve the linear quotient
                final_zero = solve_linear(quotient)
                if step_by_step:
                    print(f"✓ Remaining linear factor gives: x = {final_zero}")

                return [(candidate, 1), (final_zero, 1)]

        # No rational zeros, use quadratic formula
        if step_by_step:
            print(f"\nNo rational zeros found for quadratic. Using quadratic formula...")
        solutions, solution_type = quadratic_formula(coefficients, verbose=step_by_step)

        if solution_type == "rational":
            return [(solutions[0], 2)]  # Repeated root
        else:
            # Irrational or complex - return as strings
            return [(solutions[0], 1, solution_type), (solutions[1], 1, solution_type)]

    # Degree >= 3: find rational zeros
    if step_by_step:
        print(f"\nTesting candidates from original possible zeros list...")

    # Reorder to test last_zero first if it exists
    test_order = possible_zeros.copy()
    if last_zero and last_zero in test_order:
        test_order.remove(last_zero)
        test_order.insert(0, last_zero)
        if step_by_step:
            print(f"Testing x = {last_zero} first (checking for multiplicity)...")

    # Test each candidate
    for candidate in test_order:
        if step_by_step:
            quotient, remainder = display_synthetic_division_interactive(coefficients, candidate)
        else:
            quotient, remainder = synthetic_division(coefficients, candidate)

        if remainder == 0:
            # Found a zero!
            if step_by_step:
                print(f"\n{'═' * 70}")
                print(f"✓ ZERO FOUND: x = {candidate}")
                print(f"{'═' * 70}")

            # Recursively find remaining zeros
            remaining_zeros = find_all_zeros_recursive(quotient, possible_zeros, step_by_step, candidate)

            # Combine results - check if this zero repeats
            combined = [(candidate, 1)]
            for zero, mult, *extra in remaining_zeros:
                if zero == candidate:
                    # Same zero found again - increase multiplicity
                    combined[0] = (candidate, combined[0][1] + mult)
                else:
                    combined.append((zero, mult, *extra))

            return combined

    # No rational zeros found
    print(f"\n{'═' * 70}")
    print("NO MORE RATIONAL ZEROS FOUND")
    print(f"{'═' * 70}")
    print("\nRemaining polynomial cannot be factored using rational numbers.")
    print("You may need to use:")
    print("  • Numerical methods")
    print("  • Graphing")
    print("  • Advanced factoring techniques")

    return []


def display_complete_factorization(original_poly, all_zeros, degree=None, possible_zeros=None):
    """
    Displays the complete factorization with all zeros found.

    Args:
        original_poly (str): Original polynomial string
        all_zeros (list): List of (zero, multiplicity) or (zero, multiplicity, type) tuples
        degree (int): Degree of the polynomial
        possible_zeros (list): List of all possible rational zeros from Rational Zero Theorem
    """
    print("\n" + "=" * 70)
    print("COMPLETE FACTORIZATION")
    print("=" * 70)

    print(f"\nOriginal polynomial: {original_poly}")

    if degree is not None:
        print(f"Degree: {degree}")

    if possible_zeros is not None:
        zeros_display_list = [str(z) for z in possible_zeros]
        print(f"\nPossible rational zeros (from Rational Zero Theorem):")
        print(f"  {', '.join(zeros_display_list)}")
        print(f"  Total candidates: {len(possible_zeros)}")

    # Build factored form
    factors = []
    zeros_display = []
    has_irrational = False
    has_complex = False

    for zero_info in all_zeros:
        if len(zero_info) == 2:
            zero, mult = zero_info
            zero_type = "rational"
        else:
            zero, mult, zero_type = zero_info

        # Build factor string
        if zero_type == "real":
            # Irrational real zero - calculate approximate decimal
            has_irrational = True
            # Parse the string to get approximate value
            import re
            import math

            # Try to extract the numeric value for approximation
            try:
                # Handle format like "(-5 + √333) / 2"
                if '√' in zero:
                    # Extract the parts
                    match = re.search(r'\(([+-]?\d+)\s*([+-])\s*√(\d+)\)\s*/\s*(\d+)', zero)
                    if match:
                        const = int(match.group(1))
                        sign = 1 if match.group(2) == '+' else -1
                        sqrt_val = int(match.group(3))
                        denom = int(match.group(4))
                        approx = (const + sign * math.sqrt(sqrt_val)) / denom
                        zeros_display.append(f"  x = {zero}  ≈ {approx:.4f} (irrational)")
                    else:
                        zeros_display.append(f"  x = {zero} (irrational)")
                else:
                    zeros_display.append(f"  x = {zero} (irrational)")
            except:
                zeros_display.append(f"  x = {zero} (irrational)")
            continue
        elif zero_type == "complex":
            # Complex zero
            has_complex = True
            zeros_display.append(f"  x = {zero} (complex)")
            continue
        else:
            # Rational zero
            if zero >= 0:
                factor = f"(x - {zero})"
            else:
                factor = f"(x + {abs(zero)})"

            if mult > 1:
                factor += f"^{mult}"

            factors.append(factor)

            # Display zero with multiplicity
            if mult > 1:
                zeros_display.append(f"  x = {zero} (multiplicity: {mult})")
            else:
                zeros_display.append(f"  x = {zero}")

    # Display factored form
    if factors:
        factored_form = " × ".join(factors)
        if has_irrational or has_complex:
            if has_irrational:
                factored_form += " × (irrational quadratic factor)"
            if has_complex:
                factored_form += " × (complex quadratic factor)"
        print(f"\nFactored form: {factored_form}")
    elif has_irrational or has_complex:
        print(f"\nFactored form: Product of irrational/complex factors")
        print(f"  (Cannot be expressed as simple rational factors)")

    # Display all zeros
    print(f"\nAll zeros found:")
    for zero_str in zeros_display:
        print(zero_str)

    print(f"\nTotal zeros: {sum(z[1] for z in all_zeros if len(z) >= 2)}")


def display_factored_form(original_poly, zero_found, quotient_coeffs):
    """
    Displays the factored form after finding a zero.

    Args:
        original_poly (str): Original polynomial as string
        zero_found (Fraction): The zero that was found
        quotient_coeffs (list): Quotient coefficients after division
    """
    print("\n" + "=" * 70)
    print("FIRST ZERO FOUND - PARTIAL FACTORIZATION")
    print("=" * 70)

    # Create the factor (x - zero)
    # If zero is positive: (x - zero)
    # If zero is negative: (x + |zero|)
    if zero_found >= 0:
        factor = f"(x - {zero_found})"
    else:
        factor = f"(x + {abs(zero_found)})"

    # Create the quotient polynomial string
    quotient_poly = display_polynomial(quotient_coeffs, show_zeros=False)

    # Handle quotient display based on degree
    if len(quotient_coeffs) > 1:
        quotient_display = f"({quotient_poly})"
    else:
        quotient_display = quotient_poly

    print(f"\nOriginal polynomial: {original_poly}")
    print(f"\nPartial factorization: {factor} × {quotient_display}")
    print(f"\nFirst zero: x = {zero_found}")
    print(f"Quotient polynomial: {quotient_poly}")
    print(f"Quotient degree: {len(quotient_coeffs) - 1}")


def display_theorem():
    """
    Displays the Rational Zero Theorem explanation.
    """
    print("""
For a polynomial with integer coefficients:

    f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀

If p/q is a rational zero (where p and q have no common factors), then:

    • p is a factor of the constant term (a₀)
    • q is a factor of the leading coefficient (aₙ)

This theorem gives us ALL POSSIBLE rational zeros to test.
Not all of these will actually be zeros - they're candidates to check!
""")


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

    # Ask user for display mode at the beginning
    print("\n" + "=" * 70)
    print("CHOOSE DISPLAY MODE")
    print("=" * 70)
    print("\n  1. Step-by-step guide (educational - shows all work)")
    print("  2. Quick answer (just calculate and show final result)")

    mode_choice = input("\nEnter choice (1 or 2): ").strip()
    step_by_step = (mode_choice == '1')

    if step_by_step:
        print("\nStep-by-step mode selected. You'll see detailed explanations.")
    else:
        print("\nQuick mode selected. Calculating final answer...")

    try:
        # Parse the polynomial
        if step_by_step:
            print("\n" + "-" * 70)
            print("STEP 1: PARSING POLYNOMIAL")
            print("-" * 70)

        coefficients, degree, has_decimals = parse_polynomial(poly_input)

        if step_by_step:
            print(f"\nOriginal input: {poly_input}")
            print(f"Parsed as: {display_polynomial(coefficients, show_zeros=True)}")
            print(f"Degree: {degree}")

            # Display coefficients with clearer explanation
            coef_display = [int(c) if c == int(c) else c for c in coefficients]
            print(f"\nCoefficients [from highest to lowest degree]: {coef_display}")

            # Show what each coefficient represents
            coef_explanation = []
            for i, c in enumerate(coef_display):
                power = degree - i
                if power > 0:
                    coef_explanation.append(f"{c}x^{power}" if power > 1 else f"{c}x")
                else:
                    coef_explanation.append(f"{c}")
            print(f"  → {' + '.join(coef_explanation).replace(' + -', ' - ')}")
            print("  (Ready for synthetic division!)")

        # Convert decimals to integers if necessary
        step_num = 2  # Track step numbers
        if has_decimals:
            if step_by_step:
                print("\n" + "-" * 70)
                print("STEP 2: CONVERTING DECIMALS TO INTEGERS")
                print("-" * 70)
                print("\nRational Zero Theorem requires integer coefficients.")
                print("Converting by multiplying through...")

            coefficients, multiplier = convert_to_integers(coefficients)

            if step_by_step:
                print(f"\nMultiplied all terms by: {multiplier}")
                print(f"Integer form: {display_polynomial(coefficients, show_zeros=True)}")
                print(f"Integer coefficients: {coefficients}")
                print("\nNote: This doesn't change the zeros of the function!")
            step_num = 3

        # Extract leading coefficient and constant term
        leading_coef = int(coefficients[0])
        constant_term = int(coefficients[-1])

        # Validate
        if leading_coef == 0:
            print("\nError: Leading coefficient cannot be zero!")
            return

        if constant_term == 0 and step_by_step:
            print("\nNote: Constant term is 0, so x = 0 is automatically a zero.")
            print("For non-zero rational zeros, we can factor out x and analyze the remaining polynomial.")

        # Display the theorem
        if step_by_step:
            print("\n" + "-" * 70)
            print(f"STEP {step_num}: RATIONAL ZERO THEOREM")
            print("-" * 70)
            display_theorem()
            step_num += 1

            # Show the calculation steps
            print("-" * 70)
            print(f"STEP {step_num}: FINDING POSSIBLE RATIONAL ZEROS")
            print("-" * 70)
            step_num += 1

            print(f"\nLeading coefficient (aₙ): {leading_coef}")
            print(f"Constant term (a₀): {constant_term}")

        # Find factors
        p_factors = find_factors(abs(constant_term))
        q_factors = find_factors(abs(leading_coef))

        if step_by_step:
            print(f"\nFactors of constant term ({constant_term}): ±{p_factors}")
            print(f"Factors of leading coefficient ({leading_coef}): ±{q_factors}")

        # Calculate rational zeros
        possible_zeros = calculate_rational_zeros(leading_coef, constant_term)

        if step_by_step:
            print(f"\nPossible rational zeros (p/q):")
            print("-" * 70)

            # Display as fractions
            zeros_display = [str(z) for z in possible_zeros]
            print(", ".join(zeros_display))

            print(f"\nTotal possible rational zeros: {len(possible_zeros)}")

            # Show testing step header
            print("\n" + "=" * 70)
            print(f"STEP {step_num}: TESTING CANDIDATES WITH SYNTHETIC DIVISION")
            print("=" * 70)
            print("\nTesting candidates one by one...\n")
        else:
            print("\nFinding zeros...")

        # In quick mode, automatically find all zeros without intermediate display
        # In step-by-step mode, show the process
        original_poly_str = display_polynomial(coefficients, show_zeros=False)
        int_coeffs = [int(c) for c in coefficients]

        if not step_by_step:
            # Quick mode: just find all zeros automatically
            all_zeros = find_all_zeros_recursive(int_coeffs, possible_zeros, step_by_step=False, last_zero=None)

            if all_zeros:
                print("\n" + "=" * 70)
                print("RESULT")
                print("=" * 70)
                display_complete_factorization(original_poly_str, all_zeros, degree, possible_zeros)
            else:
                print("\n" + "=" * 70)
                print("NO RATIONAL ZEROS FOUND")
                print("=" * 70)
                print("\nThis polynomial has no rational zeros.")
                print("It may have irrational or complex zeros only.")
        else:
            # Step-by-step mode: show first zero, then ask to continue
            zero_found = None
            quotient_result = None

            for candidate in possible_zeros:
                # Perform synthetic division with interactive step-through
                quotient, remainder = display_synthetic_division_interactive(coefficients, candidate)

                # Check if we found a zero
                if remainder == 0:
                    zero_found = candidate
                    quotient_result = quotient
                    break  # Stop after finding first zero

                # If not a zero, ask if they want to continue testing
                if remainder != 0:
                    continue_testing = input("\nContinue testing next candidate? (y/n): ").strip().lower()
                    if continue_testing != 'y':
                        print("\nStopping synthetic division tests.")
                        break
                print()

            # Display results of first zero
            if zero_found is not None:
                # Show the first zero found
                display_factored_form(original_poly_str, zero_found, quotient_result)

                # Ask if user wants to find ALL zeros
                print("\n" + "=" * 70)
                print("CONTINUE FACTORING TO FIND ALL ZEROS?")
                print("=" * 70)

                print("\nWe found the first zero. We can now:")
                print("  1. Find ALL remaining zeros automatically (recommended)")
                print("  2. Stop here")

                continue_all = input("\nFind all zeros automatically? (y/n): ").strip().lower()

                if continue_all == 'y':
                    # Use recursive function to find ALL zeros
                    print("\n" + "=" * 70)
                    print("FINDING ALL ZEROS")
                    print("=" * 70)

                    all_zeros = find_all_zeros_recursive(int_coeffs, possible_zeros, step_by_step, last_zero=None)

                    # Display complete factorization
                    display_complete_factorization(original_poly_str, all_zeros, degree, possible_zeros)
                else:
                    print("\nFactoring stopped at first zero.")
                    print(f"Quotient remaining: {display_polynomial(quotient_result, show_zeros=False)}")
                    print(f"\nTo continue, run the program again with the quotient polynomial.")

            else:
                # No zeros found
                print("\n" + "=" * 70)
                print("NO RATIONAL ZEROS FOUND")
                print("=" * 70)
                print("\nNone of the possible rational zeros are actual zeros!")
                print("\nThis means the polynomial either:")
                print("  • Has only irrational zeros (like √2, √3, etc.)")
                print("  • Has only complex zeros (involving i)")
                print("  • Cannot be factored using rational numbers")
                print("\nYou may need to use:")
                print("  • Quadratic formula (if degree 2)")
                print("  • Numerical methods")
                print("  • Graphing to approximate zeros")

    except ValueError as e:
        print(f"\nError parsing polynomial: {e}")
        print("Please check your input format.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
