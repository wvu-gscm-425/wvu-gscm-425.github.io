---
title: "03 | Functions, Flow Control, & Conditionals"
subtitle: "Designing Reusable Code, Looping, & Decision Making"
format: html
order: 3
---

## Overview

So far, we’ve used Python’s built-in functions. In this module, you’ll learn to create and run your own functions, and see how one function can call another. We’ll also introduce the `for` loop to repeat computations and the if statement to execute different code based on program state. Finally, we’ll learn how to handle external files, update variables, and perform searches in strings.

## Defining new functions

A **function definition** specifies the name of a new function and the sequence of statements that run when the function is called:

```{python}
def print_lyrics():
    print("Almost Heaven, West Virginia")
    print("Blue Ridge Mountains, Shenandoah River")
```

`def` is a keyword that indicates a function definition. The name of this function is `print_lyrics`. The empty parentheses after the name indicate that this function doesn't take any arguments.

The first line of the function definition is the **header**, and the rest is the **body**. The header must end with a colon, and the body must be indented (by convention, four spaces). The body of this function is two `print()` statements, but in general, the body can include any number of statements.

Defining a function creates a **function object**, which you can display:

```{python}
print_lyrics
```

The output shows that `print_lyrics` takes no arguments. `__main__` is the name of the module that contains `print_lyrics`.

You can call this function the same way as any other Python function:

```{python}
print_lyrics()
```

## Parameters

Some of the functions we've seen require arguments: `abs()` takes a number, and `math.pow()` takes two arguments (the base and the exponent). We can make our own:

```{python}
def print_twice(string):
    print(string)
    print(string)
```

The variable in parentheses is called a **parameter**. When the function is called, the value of the argument is assigned to the parameter. For example:

```{python}
print_twice("Hail WV!")
```

This has the same effect as assigning the argument to the parameter and then executing the body:

```{python}
string = "Hail WV!"
print(string)
print(string)
```

You can also pass a variable as an argument:

```{python}
line = "Hail WV!"
print_twice(line)
```

## Calling functions

Once you define a function, you can use it inside another. Here’s a playful example of printing lyrics for ["Turn Down For What" by DJ Snake & Lil Jon](https://genius.com/Dj-snake-and-lil-jon-turn-down-for-what-lyrics):

```
Verse:
Fire up that loud
Another round of shots

Chorus:
Turn down for what?
Turn down for what?
Turn down for what?
Turn down for what?
Turn down for what?

Build:
shots! shots! shots! shots!
shots! shots! shots! shots!
shots! shots! shots! shots!
shots! shots! shots! shots!
```

As Lil Jon suggests, we need to buy a lot of rounds of shots. We can start with a helper function:

```{python}
def repeat(word, n):
    print(word * n)
```

```{python}
repeat("shots! ", 3)
```

```{python}
def print_build():
    repeat("shots! ", 4)
    repeat("shots! ", 4)
    repeat("shots! ", 4)
    repeat("shots! ", 4)

print_build()
```

`print_build()` calls `repeat()`, which then calls `print()`. We could do the same with fewer functions, but this illustrates how functions can work together.

If we want to control how many times `repeat()` repeats, we add a parameter to `print_build()`:

```{python}
def print_build(n):
    repeat("shots! ", n)
    repeat("shots! ", n)
    repeat("shots! ", n)
    repeat("shots! ", n)
```

Then we can call:

```{python}
print_build(4)
```

Next, let’s add verse and chorus functions:

```{python}
def print_verse():
    print("Fire up that loud")
    print("Another round of shots")

print_verse()
```

```{python}
def print_chorus():
    print("Turn down for what?")
    print("Turn down for what?")
    print("Turn down for what?")
    print("Turn down for what?")
    print("Turn down for what?")

print_chorus()
```

Now bring it all together:

```{python}
print_verse()
print_chorus()
print_verse()
print_chorus()
print_build(4)
```

We’re repeating some lines of code explicitly, which isn’t ideal. We’ll address that soon.

## Repetition

To print something multiple times, you can use a `for` loop. Here’s a simple example:

```{python}
for i in range(2):
    print(i)
```

`range(2)` creates a sequence of two values: `0` and `1`. The loop assigns each value to `i` and then runs the body. When the sequence ends, the loop ends.

Here’s a loop that prints the verse twice:

```{python}
for i in range(2):
    print("Verse", i)
    print_verse(),
    print() # adds a blank line
```

A `for` loop can appear inside a function, such as this one that prints the verse `m` times:

```{python}
def print_m_verse(m):
    for i in range(m):
        print_verse()

print_m_verse(4)
```

In this example, we don't use `i` in the body of the loop, but there has to be a variable in the header anyway.

## Variables and parameters are local

A variable created inside a function is **local**, meaning it only exists inside that function. Here’s an example:

```{python}
def cat_twice(part_1, part_2):
    cat = part_1 + part_2
    print_twice(cat)
```

```{python}
line_1 = "Country roads, "
line_2 = "take me home."
cat_twice(line_1, line_2)
```

Inside `cat_twice()`, cat is created. Outside of it, `cat` doesn’t exist:

```{python}
#| error: true
print(cat)
```

Parameters are also local. Outside `cat_twice()`, `part_1` and `part_2` don’t exist.

## Tracebacks

When a runtime error occurs inside a function, Python shows a **traceback**, listing the function that was running, the function that called it, and so on, up the “stack.” Here’s a `print_twice()` that tries to print `cat`, which is a local variable in a different function:

```{python}
def print_twice(string):
    print(cat)              # NameError
    print(cat)
```

```{python}
#| error: true
cat_twice(line_1, line_2)
```

The traceback shows that `cat_twice()` called `print_twice()`, which caused the error.

## Refactoring

Let’s reorganize our "Turn Down for What" example to avoid repeated code. This is called **refactoring**.

```{python}
# Original
def print_build(n):
    repeat("shots! ", n)
    repeat("shots! ", n)
    repeat("shots! ", n)
    repeat("shots! ", n)

# Improved
def print_build(repeats, shots):
    for i in range(repeats):
        repeat("shots! ", shots)

print_build(4, 4)
```

```{python}
# Original
def print_verse():
    print("Fire up that loud")
    print("Another round of shots")

# Improved
def print_verse(lines):
    for i in range(lines):
        print("Fire up that loud")
        print("Another round of shots")

print_verse(4)
```

```{python}
# Original
def print_chorus():
    print("Turn down for what?")
    print("Turn down for what?")
    print("Turn down for what?")
    print("Turn down for what?")
    print("Turn down for what?")

# Improved
def print_chorus(lines):
    for i in range(lines):
        print("Turn down for what?")

print_chorus(6)
```

Putting it all together:

```{python}
print_verse(1)
print()
print_chorus(5)
print()
print_verse(1)
print()
print_chorus(5)
print()
print_verse(4)
print()
print_build(6, 4)
print()
print_chorus(5)
```

Refactoring improves the code’s structure without changing its behavior. If we had planned the structure from the start, we might have avoided this step, but sometimes you only see a better design after you start coding.

## Why functions?

- **Readability**: Naming groups of statements makes code easier to read and debug.
- **Reusability**: Functions eliminate repetitive code. Changes become easier to manage.
- **Modularity**: Breaking down a program into functions lets you debug parts individually.
- **Reuse**: Well-designed functions can be used by other programs.

Wrapping code in a function is called **encapsulation**. One advantage is that a name serves as documentation. Another is that calling a function is more concise than copying and pasting its body.

Adding parameters to a function is called **generalization**, because it makes the function more general – for example, printing `"shots!"` any number of times.

When a function has several numerical arguments, it’s easy to mix them up. You can use **keyword arguments** to specify each argument by name:

```{python}
def print_build(repeats, shots):
    for i in range(repeats):
        repeat("shots! ", shots)

print_build(repeats=8, shots=4)
print()
print_build(shots=4, repeats=8)
```

## Docstrings

A **docstring** is a string at the start of a function that explains its interface:

```{python}
def print_build(repeats, shots):
    """Prints "shots" for a custom amount of times and lines

    repeats: number of lines of lyrics
    shots: number of times "shots" is printed per line
    """
    for i in range(repeats):
        repeat("shots! ", shots)
```

A good docstring explains what the function does and the effect of each parameter, without diving into internal details.

## `if` statements

**Conditional statements** let you check conditions and change the program’s behavior. The simplest form is the `if` statement:

```{.python}
if x > 0:
    print('x is positive')
```

The boolean expression after `if` is called the condition. If it’s true, Python executes the indented block; otherwise, it skips it.

If you need a block that does nothing, use `pass`:

```{.python}
if x < 0:
    pass          # TODO: need to handle negative values!
```

## Boolean expressions and logical operators

A boolean expression is either `True` or `False`. For instance:

```{python}
5 == 5
```

```{python}
5 == 7
```

The double equal sign `==` compares two values for equality.

::: {.callout-tip}
A common error is to use a single equal sign (`=`) instead of a double equal sign (`==`). Remember that `=` assigns a value to a variable and `==` compares two values.
:::

Other relational operators include:

```{.python}
x != y # x is not equal to y
x > y  # x is greater than y
x < y  # x is less than to y
x >= y # x is greater than or equal to y
x <= y # x is less than or equal to y
```

You can combine boolean expressions with **logical operators**: `and`, `or`, and `not`. For example:

```{.python}
x > 0 and x < 10
x % 2 == 0 or x % 3 == 0
not x > y
```

## The `else` clause 

`if` can include an `else` clause:

```{.python}
if x % 2 == 0:
    print('x is even')
else:
    print('x is odd')
```

One branch runs if the condition is true, the other if it’s false.

## Chained conditionals

When you have more than two possibilities, use `elif`:

```{.python}
if x < y:
    print('x is less than y')
elif x > y:
    print('x is greater than y')
else:
    print('x and y are equal')
```

Conditions are checked in order, and only the first true branch runs.

## Nested Conditionals

One conditional can be nested within another, but it can be harder to read:

```{.python}
if 0 < x:
    if x < 10:
        print('x is a positive single-digit number.')
```

Logical operators often simplify nested conditionals:

```{.python}
if 0 < x and x < 10:
    print('x is a positive single-digit number.')
```

For this kind of condition, Python provides a more concise option:

```{.python}
if 0 < x < 10:
    print('x is a positive single-digit number.')
```

## Some functions have return values

Functions like `abs`, `round`, `math.sqrt`, and `math.pow` return a value. You can assign that value to a variable or use it in an expression:

```{python}
import math
radius = math.sqrt(42 / math.pi)
area = math.pi * radius**2
```

You can also write your own function with a return value:

```{python}
def circle_area(radius):
    area = math.pi * radius**2
    return area
```

```{python}
a = circle_area(radius)
a
```

However, local variables inside a function (like `area`) don’t exist outside that function.

```{python}
#| error: true
area
```

## And some have `None`

If a function doesn’t use `return`, it returns `None`, a special value:

```{python}
def repeat(word, n):
    print(word * n)
```

This function uses the `print` function to display a string, but it does not use a `return` statement to return a value. If we assign the result to a variable, it displays the string anyway.

```{python}
result = repeat('Shots! ', 3)
print(result)  # Displays None
```

If you want a function that returns a string rather than prints it, you can do:

```{python}
def repeat_string(word, n):
    return word * n
```

Notice that we can use an expression in a return statement, not just a variable.

## Return values and conditionals

A function can have multiple return statements, such as a reimplementation of `abs`:

```{python}
def absolute_value(x):
    if x < 0:
        return -x
    else:
        return x
```

Make sure thath every possible path hits a return statement:

```{python}
def absolute_value_wrong(x):
    if x < 0:
        return -x
    if x > 0:
        return x
    # If x is 0, returns None (missing a final else branch)
```

## Boolean functions

Functions can return the boolean values `True` and `False`, which is often convenient for encapsulating a complex test in a function. For example, `is_divisible` checks whether x is divisible by y with no remainder.

```{python}
def is_divisible(x, y):
    if x % y == 0:
        return True
    else:
        return False
```

```{python}
is_divisible(6, 4)
```

```{python}
is_divisible(6, 3)
```

Inside the function, the result of the `==` operator is a boolean, so we can write the function more concisely by returning it directly:

```{python}
def is_divisible(x, y):
    return x % y == 0
```

Boolean functions are often used in conditional statements:

```{python}
if is_divisible(6, 2):
    print('divisible')
```

It might be tempting to write something like this:

```{python}
if is_divisible(6, 2) == True:
    print('divisible')
```

But the comparison is unnecessary.

## Input validation

We now have all the tools we need to make sure that the programs we write will be executed fully, regardless of user error.

Remember, if the end use can mess up your instructions they will. Sound familiar?

Let's go back to the example of calculating the volume of a sphere with a succinct function using what we've learned so far:

```{python}
def volume_of_sphere(radius):
    from math import pi
    return (4/3) * pi * radius**3
```

If radius is an integer or float, no problem. But if we get a string, we will get an error:

```{python}
#| error: true

print(volume_of_sphere(4))
print(volume_of_sphere("4"))
```

While the default error message gives the user an idea of what's wrong, we can handle this more gracefully:

```{python}
def volume_of_sphere(radius):
    if type(radius) == int or type(radius) == float:
        from math import pi
        return (4/3) * pi * radius**3
    else:
        print("Input was not a number, try again.")
        return None

volume_of_sphere("5")
```

If this were a standalone program, it would return a result instead of stopping execution with an error.

## Debugging

Debugging can be frustrating, but it is also challenging, interesting, and sometimes even fun. And it is one of the most important skills you can learn.

In some ways debugging is like detective work. You are given clues and you have to infer the events that led to the results you see.

Debugging is also like experimental science. Once you have an idea about what is going wrong, you modify your program and try again. If your hypothesis was correct, you can predict the result of the modification, and you take a step closer to a working program. If your hypothesis was wrong, you have to come up with a new one.

For some people, programming and debugging are the same thing; that is, programming is the process of gradually debugging a program until it does what you want. The idea is that you should start with a working program and make small modifications, debugging them as you go.

If you find yourself spending a lot of time debugging, that is often a sign that you are writing too much code before you start tests. If you take smaller steps, you might find that you can move faster.

---

When a syntax or runtime error occurs, the error message contains a lot of information, but it can be overwhelming. The most useful parts are usually:

- What kind of error it was, and
- Where it occurred.

Syntax errors are usually easy to find, but there are a few gotchas. Errors related to spaces and tabs can be tricky because they are invisible and we are used to ignoring them.

```{python}
#| error: true
x = 5
 y = 6
```

In this example, the problem is that the second line is indented by one space. But the error message points to `y`, which is misleading. Error messages indicate where the problem was discovered, but the actual error might be earlier in the code.

The same is true of runtime errors. For example, suppose you are trying to convert a ratio to decibels, like this:

```{python}
#| error: true
import math
numerator = 9
denominator = 10
ratio = numerator // denominator
decibels = 10 * math.log10(ratio)
```

The error message indicates line 5, but there is nothing wrong with that line. The problem is in line 4, which uses integer division instead of floating-point division – as a result, the value of `ratio` is `0`. When we call `math.log10`, we get a `ValueError` with the message `math domain error`, because `0` is not in the "domain" of valid arguments for `math.log10`, because the logarithm of `0` is undefined.

In general, you should take the time to read error messages carefully, but don’t assume that everything they say is correct.

---

Breaking a large program into smaller functions creates natural checkpoints for debugging. If a function is not working, there are three possibilities to consider:

- There is something wrong with the arguments the function is getting – that is, a precondition is violated.
- There is something wrong with the function – that is, a postcondition is violated.
- The caller is doing something wrong with the return value.

To rule out the first possibility, you can add a `print` statement at the beginning of the function that displays the values of the parameters (and maybe their types). Or you can write code that checks the preconditions explicitly.

::: {.callout-tip}
This is a very basic version of [logging](https://do4ds.com/chapters/sec1/1-4-monitor-log.html).
:::

If the parameters look good, you can add a print statement before each return statement and display the return value. If possible, call the function with arguments that make it easy check the result.

If the function seems to be working, look at the function call to make sure the return value is being used correctly – or used at all!

Adding `print` statements at the beginning and end of a function can help make the flow of execution more visible for testing. For example, here is a version of `volume_of_sphere` with print statements:

```{python}
def volume_of_sphere(radius):
    if type(radius) == int or type(radius) == float:
        print(f"Radius is a valid type: {type(radius)}") # Showing radius is fine
        from math import pi
        return (4/3) * pi * radius**3
    else:
        print("Input was not a number, try again.") # In original example, showing radius was not fine
        return None
```

```{python}
vol = volume_of_sphere(5)
print(vol)
```

```{python}
vol = volume_of_sphere("5")
vol
```

## Exercises

### Right-Align Text

Write a function named `print_right()` that takes a string named `text` as a parameter and prints the string with enough leading spaces that the last letter of the string is in the 40th column of the display.

Hint: Use `len()`, `+`, and `*`.

Here's an example output:

```{.python}
print_right("Monty")
print_right("Python's")
print_right("Flying Circus")
```

```
                                   Monty
                                Python's
                           Flying Circus
```

### Draw a Triangle

Write a function called `triangle` that takes a string and an integer, then draws a pyramid of the given height using copies of the string. For example:

```{.python}
triangle("L", 5)
```

```
L
LL
LLL
LLLL
LLLLL
```

### Draw a Rectangle

Write a function called `rectangle` that takes a string and two integers, then draws a rectangle of the given width and height using copies of the string. For example:

```{.python}
rectangle("[]", 5, 4)
```

```
[][][][][]
[][][][][]
[][][][][]
[][][][][]
```

### Triangle Tester

Write a function named `is_triangle` that takes three integers as arguments and returns True or False depending on whether you can form a triangle with those lengths. Use the rule:

> If any of the three lengths is greater than the sum of the other two, then you cannot form a triangle. Otherwise, you can. (If the sum of two lengths equals the third, they form what is called a “degenerate” triangle.)

*Hint*: Use a chained conditional.

### Check "Between"

Write a boolean function `is_between(x, y, z)`, that returns `True` if $x < y < z$ or if $z < y < x$, and `False` otherwise.
