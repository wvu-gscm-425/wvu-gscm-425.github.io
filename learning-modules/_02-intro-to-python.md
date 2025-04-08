---
title: "02 | Intro to Python"
subtitle: "Learn about the basics of Python."
format: html
order: 2
---

## Overview

This module covers the Python basics you need for this course. It isn’t an exhaustive guide to Python, but it’s enough for anyone with previous coding experience (see [Prerequisite Course(s)](../index.qmd#course-info)).

You will learn how Python represents numbers, letters, and words, followed by arithmetic operations. We’ll also build important programming vocabulary. I won’t test you on these terms, but you’ll need them for future lessons.

Next, we’ll learn about variables, statements, the `import` statement, and the `print()` function. We’ll also discuss function arguments and Python modules.

## Arithmetic operators

An **arithmetic operator** is a symbol that represents a computation. For example:

- `+` performs addition:

```{python}
2000 + 25
```

- `-` performs subtraction:

```{python}
2030 - 5
```

- `*` performs multiplication:

```{python}
405 * 25
```

- `/` performs division:

```{python}
10125 / 5
```

Notice that the result of division is `42.0` vs `42`. Python recognizes two numeric types:

1. **integers**: numbers without a decimal part
2. **floating-point numbers (float)**: numbers with a decimal point (including integer-like values stored in floating form)

If you add, subtract, or multiply two integers, the result remains an integer. However, dividing two integers produces a floating-point result. 

Python also supports **integer division** with the operator `//`, which always return an integer:

```{python}
4050 // 2
```

This operator is called "floor division" because it always rounds down:

```{python}
4051 // 2
```

The **modulus** operator `%` returns the remainder after dividing two numbers:

```{python}
4051 % 2 # remainder is 1
```

If a number divides evenly, `%` returns `0`:

```{python}
4050 % 2 # remainder is 0
```

Finally, `**` performs exponentiation (raising a number to a power):

```{python}
4.58442641 ** 5
```

::: {.callout-tip}
In other languages, like **R**, you use the caret `^` for exponentiation, but in Python `^` is the "XOR" operator, which we won’t cover here.
:::

## Expressions

An **expression** is a combination of operators and values:

```{python}
6 + 6 ** 2
```

Python follows standard order of operations:

```{python}
12 + 5 *6
```

Use parentheses to change that order:

```{python}
(12 + 5) * 6
```

Every expression evaluates to a **value**, so `6 * 7` becomes `42`.

## Arithmetic functions

Python provides **functions** that work with numbers, such as:

- `round()` rounds a float to the nearest integer:

```{python}
round(4.58442641 ** 5)
```

- `abs()` returns the absolute value

```{python}
abs(-2025)
```

When you **call** a function, you must use parentheses. Omitting them causes a syntax error:

```{python}
#| error: true

abs 42 # correct usage: abs(42)
```

If you type only the function name:

```{python}
abs
```

Python tells you that `abs` is indeed a function, along with some extra details.

## Strings

A **string** is a sequence of characters. You can enclose them in single or double quotes:

```{python}
print('Hello')
print("World")
```

Use double quotes if you need an apostrophe, or single quotes if you need a double quote:

```{python}
print("That's her book")
print('"I see," he said')
```

::: {.callout-tip}
Python treats single and double quotes the same; you can choose either as long as you’re consistent.
:::

**Triple-quoted strings** can span multiple lines or contain both single and double quotes:

```{python}
print('''"That's great", she said.''')

print("""
To be,
or not to be,
that is definitely a question.
""")
```

Strings can hold spaces, punctuation, and digits:

```{python}
print("How the turn tables... uhh wait. What was line 5?")
```

Use the `+` operator to **concatenate** (join) strings:

```{python}
print('Well, ' + "it's a small " + 'world.')
```

Use the `*` operator to repeat strings:

```{python}
print('RA' + 'TA' * 3)
```

Other arithmetic operators don't work on strings.

You can use `len()` to find a string’s length:

```{python}
len('12345')
```

::: {.callout-note}
Notice that `len()` counts the the letters between the quotes, but not the quotes themselves.
:::

Avoid backticks `` ` `` or curly quotes `“ ”` because they cause syntax errors:

```{python}
#| error: true

print(`hello`)
```

```{python}
#| error: true

print(“hello”)
```

::: {.callout-tip}
Most code editors color valid strings differently, so keep an eye on syntax highlighting to avoid mistakes.
:::

## Values and types

We’ve encountered three kinds of values:

1. `2` (integer)
2. `42.0` (float)
3. `"text"` (string)

A kind of value is called a **type**. Every value has a type, also referred to as "belongs to" a type.

Python provides a function, `type()`, that tells you the type of any value:

```{python}
type(2)
```

```{python}
type(42.0)
```

```{python}
type("text")
```

`int`, `float`, and `str` can also convert values:

```{python}
int(42.9) # rounds down to 42
```

```{python}
float(42) # converts integer 42 to float 42.0
```

```{python}
str(123) # converts number 123 to the string "123"
```

If you try arithmetic on a string, you get an error:

```{python}
print(123 * 3)   # numeric multiplication
print("123" * 3) # string repetition
```

```{python}
#| error: true

print("500" / 5) # TypeError: can't divide a string by an integer
```

To fix this, **cast** to the right type:

```{python}
int("126") * 3
```

If you have a large integer, you may type it like `1,000,000`. This is a legal expression in Python, but the value is not what you would expect:

```{python}
1,000,000
```

Python interprets `1,000,000` as a comma-separated sequence of integers. We'll learn about this kind of sequence later.

You can use underscores to make large numbers easier to read:

```{python}
1_000_000
```

## Formal and natural languages

Natural languages (English, Spanish, etc.) evolved over time and rely on context, idioms, and sometimes ambiguity. Formal languages like Python are precise and unambiguous. Python does exactly what you write, so details matter. Small mistakes in spelling or punctuation can cause big errors. You might find this rigid at first, but you’ll adapt with practice.

## Variables

A **variable** is a name that refers to a value. Create one with an **assignment statement**:

```{python}
n = 17
```

The assignment has three parts:

1. The variable name
2. The `=` operator
3. An expression (`17` here)

```{python}
pi = 3.141592653589793
message = "pie > π"
```

Once assigned, you can use these variables:

```{python}
print(message)
print(n + 5)
print(2 * pi)
print(round(pi))
print(len(message))
```

## Variable names

You can use letters and digits in a variable name but cannot start with a digit. Although uppercase letters are allowed, most Python code uses lowercase. Use underscores to connect words: `your_name` or `airspeed_of_unladen_swallow`.

A name containing punctuation (`million!`) or starting with a number (`76trombones`) triggers a syntax error. Some words, like `class`, are keywords and cannot be variable names.

```{python}
#| error: true

million! = 1000000
```

`76trombones` is illegal because it starts with a number.

```{python}
#| error: true

76trombones = 'big parade'
```

`class` is also illegal, but it might not be obvious why.

```{python}
#| error: true

class = 'Defense Against the Dark Arts'
```

::: {.callout-tip}
Your editor will often highlight keywords in a different color so you can recognize them.
:::

## The `import` statement

Some features require **importing**. For example, to use the `math` module:

```{python}
import math
```

A **module** is a collection of variables and functions. [Python's math module](https://docs.python.org/3/library/math.html) provides a variable called `pi` that contains the value of the mathematical constant π:

```{python}
math.pi
```

Use the **dot** to access module features:

```{python}
print(math.sqrt(25))
print(math.pow(5, 2)) # 5 ** 2 behaves the same as math.pow(5, 2)
```

## Expressions vs statements

An **expression** calculates a value, regardless of its complexity:

```{python}
19 + n + round(math.pi) * 2
```

A **statement** performs an action without producing a value you can use:

```{python}
n = 17
```

We **evaluate** expressions to get their value and **execute** statements to perform actions.

## The `print()` function

When you type an expression in many Python environments, it displays the result. But if you have multiple expressions in a single cell (or script), only the last one appears. Use `print()` to display more than one item:

```{python}
print(n + 2)
print(n + 3)
print("The value of pi is approximately", math.pi)
```

::: {.callout-note}
`print()` separates arguments with a space by default.
:::

## Function Arguments

The expressions inside a function call’s parentheses are **arguments**. Different functions accept different numbers of arguments:

- `int()` can take one required argument and an optional base:

```{python}
int("101", 2)
```

- `math.pow()` takes two arguments:

```{python}
math.pow(5, 2)
```

- `round()` can take an optional second argument (decimals to round):

```{python}
round(math.pi, 3)
```

- `print()` accepts any number of arguments:

```{python}
print("Any", "number", "of", "arguments")
```

If you supply too many or too few arguments, or if the arguments are the wrong type, Python raises a TypeError:

```{python}
#| error: true

float("123", 2)
```

```{python}
#| error: true

math.pow(2)
```

```{python}
#| error: true

math.sqrt("25")
```

## Comments

As code grows, **comments** clarify your reasoning or record important details. Python ignores text after a `#`:

```{python}
# Variables
force = 500 # Force in Newtons (N)
area = 0.01 # Cross-sectional area in square meters (m²)

# Formula
stress = force / area

# Output
print("Stress:", stress, "in (Pa)")
```

**Good comments** explain why you wrote the code in a certain way (especially if it’s not obvious):

```{python}
# Use SI units for consistency throughout the program
area = 0.01
```

Bad comments restate the obvious:

```{python}
area = 0.01  # set area to 0.01
```

Well-chosen variable names can reduce the need for comments, but avoid names so long that they make expressions unreadable.

## Debugging

We call mistakes in code bugs, and the process of finding and fixing them debugging. You might feel frustrated when things break, but remember it's normal and part of learning.

Think of the computer as precise but inflexible. You must provide exact instructions because it doesn't infer or guess what you mean.

### Common Errors

1. **Syntax erros**: The code violates Python’s structure rules. Python refuses to run the code and points out where it got stuck:

```{python}
#| error: true

million! = 1000000
```

2. **Runtime errors**: The code starts running but fails during execution, like dividing a string by an integer:

```{python}
#| error: true

"126" / 3
```

3. **Semantic errors**: The code runs but does something unintended. For example, mixing up units might produce the wrong numeric result without an obvious error:

```{python}
# This uses area in cm² instead of m²
force = 500       # Newtons
area = 10         # cm² (wrong units)
stress = force / area
print(stress)     # No error, but incorrect value => semantic error
```

## Exercises

### Rounding Behavior

Experiment with `round()` when a number ends in `0.5`. It sometimes rounds up and sometimes down. Figure out the pattern:

```{python}
round(42.5)
```

```{python}
round(43.5)
```

### Deliberate Mistakes

a. What happens if you use a `+` sign before a number (`+2`) or repeat it (`2++2`)? What about `2--2`?
b. What if you write two values without an operator (`4 2`)?
c. What if you call `round(42.5)` but remove one or both parentheses?

### Type Checking

Guess each expression's type and then use `type()` to check:

- `765`
- `2.718`
- `"2 pi"`
- `abs(-7)`
- `abs`
- `int`
- `type`

### More Arithmetic

- How many seconds are there in 42 minutes 42 seconds?
- How many miles are there in 10 kilometers (`1.61 kilometers/mile`)? 
- If you run a 10 kilometer race in 42 minutes 42 seconds, what is your average pace in seconds per mile?

### Intentional Errors

- We've seen `n = 17` is legal, what about `17 = n`?
- What about `x = y = 1`?
- In some languages every statement ends with a semi-colon (`;`). What happens if you put a semi-colon at the end of a Python statement?
- What if you put a period at the end of a statement?
- What happens if you spell the name of a module wrong and try to `import maath`?

### Volume of a Sphere

The formula for the volume of a sphere with radius $r$ is $\frac{4}{3} \pi r^3$. Compute it for $r = 5$.

- Start with a variable names `radius` (in centimeters)
- Compute `volume` (in cubic centimeters)
- Print the result and include comments

### Trigonometry

According to a trig identity, $(\cos x)^2 + (\sin x)^2 = 1$. Test this for `x = 42`.

- Create a variable named `x` to `42`
- Use `math.cos()` and `math.sin()` to compute

::: {.callout-note}
It might not be exactly 1 because of floating-point imprecision. 
:::

### Exploring $e$

The `math` module defines `e`. Compute $e^2$ in three ways:

- Use `math.e` and the exponentiation operator.
- Use `math.pow` to raise `math.e` to the power `2`.
- Use `math.exp`, which takes as an argument a value, $x$, and computes $e^x$.

Compare the results and see which appears most accurate.
