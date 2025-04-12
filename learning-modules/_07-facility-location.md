---
title: "07 | Facility Location Models" 
subtitle: "Optimizing Facility Placement in Supply Chain Networks" 
format: 
  html: 
    other-links: 
      - text: "Customer Data (20 locations)" 
        href: "data/multi_city_data.csv"
        icon: file-earmark-spreadsheet
      - text: "Exercise Data" 
        href: "data/multi_city_data_2.csv"
        icon: file-earmark-spreadsheet
order: 7
---

## Overview

Facility location problems involve finding the best positions for one or more facilities (e.g. warehouses, plants, service centers) to minimize costs or improve service in a supply chain. These decisions are critical because facility locations impact transportation costs, response times, and customer service levels. In this module, you will learn how to:

1. Formulate single-facility location problems (one new facility) to minimize transportation cost, using Euclidean distance-based objectives.
2. Extend to multi-facility location problems (multiple new facilities) where customer demand points must be assigned to the nearest facility.
3. Understand nonlinear distance-based formulations (using Euclidean distance) and how to handle them in optimization (using squared distances for tractability).
4. Implement and solve location models in Python with Gurobi, using simple data structures (lists/dictionaries) and loops for clarity.
5. Interpret solutions and discuss trade-offs, including diminishing returns from adding more facilities.
6. Throughout, we include conceptual questions and hands-on Python exercises to reinforce understanding. By the end, you will be able to model facility location decisions and solve them using optimization techniques.

Throughout, we include conceptual questions and hands-on Python exercises to reinforce understanding. By the end, you will be able to model facility location decisions and solve them using optimization techniques.

## Key Concepts in Facility Location

**Facility location** decisions aim to position facilities such that **transportation costs or distances** are minimized (or service metrics optimized). Key concepts include:

- **Demand points (customers)**: Locations that require service (with given coordinates on a plane and demand volume).
- **Facility (warehouse/plant) location**: The decision variables, typically the $(x, y)$ coordinates, of the facility to be located.
- **Distance and cost**: Usually, cost is proportional to distance traveled * demand. We often use Euclidean distance (straight-line distance) between a facility and a customer: $\text{Distance}_i(x, y) = \sqrt{(x - x_i)^2 + (y - y_i)^2}$, where $(x_i, y_i)$ are coordinates of customer $i$ and $(x,y)$ is the facility location.
  - Total transport cost = $\sum_{i} (\text{distance}{i}) \times (\text{demand}{i}) \times (\text{transport cost per unit distance})$.
- **Nonlinearity:** The objective involving Euclidean distance is **nonlinear** (the distance formula has a square root). This makes the problem a **non-linear programming (NLP)** problem. However, it is a convex problem for a single facility (no local minima besides the global minimum).
- **Squared distance approximation**: To use solvers like Gurobi (which handle linear and quadratic objectives), we often minimize the **squared distance**; $(x-x_i)^2 + (y-y_i)^2$ instead. Minimizing squared distance will locate the facility similarly to minimizing actual distance (it places more weight on farther distances). The optimal location for squared distance is known as the **weighted center of gravity** of the demand points.

### Single-facility vs Multi-facility

- *Single-facility location*: One facility to serve all demand points. We seek the best single location to minimize total distance costs.
- *Multi-facility location*: Placing multiple facilities (e.g. multiple warehouses) and assigning each customer to the nearest (or most appropriate) facility. This is more complex because we must decide both **facility locations** and **customer-facility assignments** simultaneously. The problem becomes a location-allocation problem, often requiring heuristic or iterative methods (as it is generally an NP-hard problem when formulated exactly).

### Why is facility location important?

Proper placement of facilities can dramatically reduce transportation costs and improve service. For example, a company may reduce delivery distances by building regional warehouses closer to customers rather than serving an entire country from one central location. However, more facilities also mean higher fixed costs (which we **do not** explicitly model here, but in practice there is a trade-off).

### Metrics

While this module focuses on minimizing total distance (or cost) to customers (a typical objective, yielding a **1-median** problem on a plane), other metrics could be optimized:

- *Minimize average distance per customer*: similar to total distance if total demand is fixed.
- *Minimize maximum distance to any customer*: a minimax or covering radius objective (placing facility to minimize the farthest customer distance).
- *Satisfy distance constraints*: e.g. ensure all customers are within a certain distance (which may require multiple facilities if one cannot cover all within the limit).

We will primarily use the total (or weighted) distance objective, which is most common in cost minimization.

## Single-Facility Location Model

In a single-facility problem, we want to find the coordinates $(x, y)$ for a new facility that minimize the sum of distances to all customers weighted by their demand (and any transport cost rate).

Mathematically:

- Let $N$ be the set of customer locations.
- For each customer $i \in N$:
  - $(x_i, y_i)$ are known coordinates.
  - $d_i$ is the demand (or shipment volume) from the facility to $i$ (e.g. annual demand).
  - $t_i$ is the transport cost per unit distance for customer $i$ (if transport cost differs by route; often $t_i$ is the same for all if using the same transport mode)
- Decision variables: $x$ and $y$ (continuous) = coordinates of the facility.

**Objective (nonlinear)**: Minimize total transport cost = $\sum_{i \in N} d_i \cdot t_i \cdot \sqrt{(x - x_i)^2 + (y - y_i)^2}$. There are no explicit constraints on $(x,y)$ (the facility can be located anywhere on the plane, unless specified otherwise).

This is a **continuous nonlinear optimization** problem. The objective function is convex in $(x,y)$, so a global optimum can be found via calculus or iterative algorithms. However, standard linear solvers cannot be used directly due to the square root.

**Using Squared Distances**: To solve with a quadratic solver (like Gurobi’s QP capabilities), we often minimize $\sum_i d_i t_i \big[(x - x_i)^2 + (y - y_i)^2\big]$ instead. This removes the square root, making it a quadratic objective. The optimal $(x,y)$ for the squared-distance objective is the weighted center-of-gravity:

$$
x^* = \frac{\sum_i d_i t_i x_i}{\sum_i d_i t_i}, \quad
y^* = \frac{\sum_i d_i t_i y_i}{\sum_i d_i t_i}
$$

::: {.callout-tip}
If transport cost per unit distance $t_i$ is uniform for all customers, it cancels out. Then $x^*, y^*$ are just demand-weighted averages of customer coordinates. This center-of-gravity is often used as a heuristic solution for the actual distance-based problem. For the true distance objective (with sqrt), the optimum tends to be somewhat closer to high-demand or far-away points than the center-of-gravity (since each additional mile hurts the objective linearly, not quadratically). In practice, the difference may be small, and the center-of-gravity provides a good starting point.
:::

### Example 1: Single Warehouse for Sunshine Tire Company

*Sunshine Tire Company* is considering building a new distribution center to serve three major clients. The clients’ locations (in an XY grid coordinate system) and annual shipment volumes are given below:

- **Toledo** – demand = 15,000 tons/year, coordinates $(1360, 1160)$.
- **Macomb** – demand = 5,000 tons/year, coordinates $(980, 1070)$.
- **Allentown** – demand = 11,000 tons/year, coordinates $(1840, 1150)$.

Transport cost per ton-mile varies by client due to different freight contracts:

- **Toledo**: $t = $0.95$ per ton-mile
- **Macomb**: $t = $1.25$ per ton-mile
- **Allentown**: $t = $0.85$ per ton-mile

**Problem**: Find the location $(x,y)$ for the new DC that minimizes total annual transportation cost to these three clients.

First, let’s formulate the model. Using squared distances for tractability, our objective is:

$$
\min \Big[
15000(0.95)\big((x - 1360)^2 + (y - 1160)^2\big) +
5000(1.25)\big((x - 980)^2 + (y - 1070)^2\big) +
11000(0.85)\big((x - 1840)^2 + (y - 1150)^2\big)
\Big]
$$

We can solve this using Python and Gurobi (as a quadratic program). We expect the optimal solution to correspond to a weighted average of the coordinates, weighted by $d_i t_i$ for each city.

#### Solving with Python (Gurobi)

We'll use `gurobipy` to set up the decision variables and quadratic objective.

```{python}
import gurobipy as gp
from gurobipy import GRB

# Data for Sunshine Tire example
clients = {
    "Toledo":    {"x": 1360, "y": 1160, "demand": 15000, "cost_per_ton_mile": 0.95},
    "Macomb":    {"x": 980,  "y": 1070, "demand": 5000,  "cost_per_ton_mile": 1.25},
    "Allentown": {"x": 1840, "y": 1150, "demand": 11000, "cost_per_ton_mile": 0.85}
}

# Create model
m = gp.Model("SunshineTire_SingleFacility")

# Decision variables: coordinates (x, y)
x = m.addVar(name="x_coord", lb=-GRB.INFINITY, ub=GRB.INFINITY)
y = m.addVar(name="y_coord", lb=-GRB.INFINITY, ub=GRB.INFINITY)

# Build quadratic objective
quad_expr = gp.QuadExpr()
for city, data in clients.items():
    xi, yi = data["x"], data["y"]
    w = data["demand"] * data["cost_per_ton_mile"]  # weight for this term
    quad_expr += w * ((x - xi)**2 + (y - yi)**2) # Add w * [ (x - xi)^2 + (y - yi)^2 ] to objective

m.setObjective(quad_expr, GRB.MINIMIZE)
m.optimize()

# Output solution
if m.Status == GRB.OPTIMAL:
    print(f"Optimal location: x = {x.X:.3f}, y = {y.X:.3f}")
    # Compute total annual transport cost using actual distances:
    total_cost = 0
    for city, data in clients.items():
        dist = ((x.X - data["x"])**2 + (y.X - data["y"])**2)**0.5  # Euclidean distance
        cost = data["demand"] * data["cost_per_ton_mile"] * dist
        total_cost += cost
        print(f"Distance to {city}: {dist:.1f}, Annual cost: ${cost:,.0f}")
    print(f"Total annual transport cost: ${total_cost:,.0f}")
```

Running this model, Gurobi will find the optimal coordinates that minimize the quadratic objective. We then calculate the total cost at that location to interpret the result.

#### Solution

The optimal coordinates are approximately **(1430, 1138)**. This point is essentially the weighted center-of-gravity of the three clients, heavily influenced by Toledo and Allentown (the larger shipments) and their relatively lower transport rates. In fact, if we compute the weighted average manually:

- Horizontal (x) coordinate: $\frac{(15000 \cdot 0.95 \cdot 1360) + (5000 \cdot 1.25 \cdot 980) + (11000 \cdot 0.85 \cdot 1840)}{(15000 \cdot 0.95) + (5000 \cdot 1.25) + (11000 \cdot 0.85)} \approx 1430$
- Vertical (y) coordinate: $\approx 1138$

This matches the solver result, confirming the center-of-gravity logic for the squared distance formulation.

- **Interpreting the cost**: The total annual cost at this optimal location is about $7.73 million. Note that being closest to Toledo (which had the cheapest per-mile cost) is not as crucial as being centrally located to balance all costs. The chosen point lies somewhat between Toledo and Allentown. Interestingly, the solution coincides roughly with Toledo’s x-coordinate and Allentown’s y-coordinate – but that may be a coincidence of data.

- **Euclidean vs Squared Objective**: If we had minimized actual distance (with the `sqrt`), the optimal point might shift slightly. For example, you could use specialized algorithms (like the Weiszfeld algorithm) to find the true distance-minimizing location (the geometric median). In this case, because Toledo and Allentown dominate, the true optimum might move a bit closer to those cities compared to the center-of-gravity. However, the difference is often minor for many practical cases. The squared-distance solution is a reasonable approximation.

- **Concept Check**: *Why might the optimal location not be exactly at one of the customer cities?* Because that would favor that one customer too much while increasing distance to others. The optimum balances all weighted distances. Only in special cases—like one customer having an overwhelmingly large demand or cost weight—would the best location coincide with that customer’s location.

#### Adding Constraints or Alternative Objectives

In some scenarios, you might have additional constraints. For example, Sunshine Tire could require that the new DC be within a certain maximum distance of each customer for service reasons. This would impose constraints like $\sqrt{(x-x_i)^2+(y-y_i)^2} \leq D_{\max}$ for each customer $i$. These are nonlinear constraints (defining a feasible region as the intersection of circles of radius $D_{\max}$ around each customer). If $D_{\max}$ is too small to cover all points, the problem becomes infeasible unless we add more facilities.

Alternatively, consider if Sunshine Tire wanted to **minimize the maximum distance** to any client (a minimax objective). The solution would try to centrally locate such that the farthest client is as close as possible. This is a different optimization criterion and might yield a different location than the min-sum solution. (Such a problem can be formulated by introducing a variable $R$ to represent the max distance and minimizing $R$ with constraints $\sqrt{(x-x_i)^2+(y-y_i)^2} \le R$ for all $i$.)

These variations highlight how objectives and constraints can change the outcome. In practice, you’d choose the formulation that best fits the company’s service targets.

**Takeaway**: For a single facility, the center-of-gravity method (weighted by demand and cost) gives a quick answer when using squared distance or as a starting point for iterative methods. If using a solver like Gurobi, we can implement the exact formulation with quadratic objective (or even the true distance using second-order cone techniques, albeit that’s more advanced). The optimal location balances the pull of all demand points according to their weight.

## Multi-Facility Location Model

Now suppose Sunshine Tire (or another company) can build **multiple facilities**. Multi-facility location problems involve deciding both *how to place several facilities and which facility serves each customer*.

This is commonly referred to as a **Location-Allocation problem**: we are effectively clustering customers to the nearest facility and finding optimal facility positions for each cluster.

**Key challenge**: The assignment of customers to facilities and the facility locations influence each other. If we knew the assignments (which customers go to facility A vs facility B, etc.), we could compute the best location for each facility (as a single-facility problem on its cluster). Conversely, if we fix facility locations, customers should be assigned to the nearest (to minimize cost). But we don't know either upfront, it's a chicken-and-egg problem, and solving it exactly in one go is difficult because it becomes a large non-linear, non-convex optimization.

In fact, an exact formulation of the multi-facility problem would involve binary assignment variables (for which facility serves which customer) and continuous location variables. This becomes a **mixed-integer non-linear program (MINLP)** that is generally hard to solve to optimality for large sets. Instead, we rely on heuristic or iterative methods. One common approach is analogous to the **k-means clustering algorithm** used in data science:

**Iterative Algorithm for Multi-Facility Location** (for a given number of facilities $K$):

1. **Initialization**: Guess initial locations for the $K$ facilities. (For example, choose $K$ customer locations as initial facility sites, or randomly distribute them, or even take the solution of the single-facility problem and perturb it for multiple centers.)
2. **Assignment step**: Assign each customer to the **nearest** facility (compute distance from each customer to each facility and pick the smallest). This creates $K$ clusters of customers.
3. **Re-optimization step**: For each cluster, recompute the best facility location for that cluster *independently*. In other words, treat each cluster as a single-facility problem and find the optimal $(x,y)$ for that facility (usually the demand-weighted centroid of that cluster, using the squared-distance approach).
4. Update the facility locations to these new coordinates.
5. Repeat steps 2–4 until assignments no longer change (i.e., it converges). This usually yields a locally optimal solution.

This heuristic is essentially the *K-means algorithm* (with demand weights, sometimes called **weighted K-means** or **K-medians** when minimizing distances). It will converge, although it might converge to a local (not global) optimum depending on the initial start. In practice, running the algorithm with multiple different initial guesses and picking the best solution can help.

### Example 2: Multiple Warehouses for *Excellent Foods Inc.*

Consider *Excellent Foods Inc.*, a company that supplies canned goods. They have 20 customer locations (retailers/wholesalers) across a region. Each customer has a yearly demand (in truckloads) and known coordinates on a grid. The transport cost is approximately $1 per unit per distance (so we can just use distance × demand as the cost). The company wants to determine where to build warehouses to minimize total distribution cost.

Initially, consider just **one warehouse** (this reduces to the single-facility problem we already solved conceptually). Then, evaluate **two**, **three**, up to **ten** warehouses to see how additional facilities reduce cost. Instead of formulating a giant MINLP, we’ll use the iterative location-allocation heuristic.

The full dataset of 20 customers with their coordinates and demands is provided in the linked spreadsheet:

:::: {.columns}

::: {.column width="45%"}

| Customer    |    Y |    X |   Demand |
|:------------|-----:|-----:|---------:|
| Customer 1  | 9.74 | 8.18 |       41 |
| Customer 2  | 8.14 | 7.46 |       62 |
| Customer 3  | 9.96 | 3.56 |       72 |
| Customer 4  | 5.13 | 6.51 |      103 |
| Customer 5  | 3.72 | 2.1  |        2 |
| Customer 6  | 7.04 | 0.28 |        3 |
| Customer 7  | 1.48 | 5.23 |        4 |
| Customer 8  | 9.72 | 4.09 |       78 |
| Customer 9  | 5.98 | 7.91 |       87 |
| Customer 10 | 0.54 | 3.2  |       94 |

:::

::: {.column width="10%"}
:::

::: {.column width="45%"}

| Customer    |    Y |    X |   Demand |
|:------------|-----:|-----:|---------:|
| Customer 11 | 5.39 | 4.22 |      118 |
| Customer 12 | 1.56 | 2.65 |        1 |
| Customer 13 | 9.45 | 8.58 |        6 |
| Customer 14 | 6.74 | 4.91 |        3 |
| Customer 15 | 2.15 | 6.34 |       81 |
| Customer 16 | 5.69 | 8.04 |      105 |
| Customer 17 | 0.8  | 7.51 |      156 |
| Customer 18 | 6.96 | 2.18 |      105 |
| Customer 19 | 6.03 | 4.35 |        2 |
| Customer 20 | 9.48 | 5.77 |        7 |

:::

::::

::: {.callout-note}
Coordinates are in some consistent unit, say tens of miles; demand is units of product shipped per year. Higher-numbered customers like 17 have large demands, in this case 156, which will influence optimal facility placement.
:::

#### Assignment and Re-optimization with Python

We'll solve this step by step for a given set of $K$ (number of warehouses). We will use the following packages to manipulate the data, run the k-means algorithm, and plot our results:

```{python}
import pandas as pd
import numpy as np
import altair as alt
from math import hypot
```

::: {.callout}
You can use the following to add them to your environment: `uv add pandas numpy altair`
:::

##### Helper Functions

First, we are going to create a few custom helper functions (use these in future exercises):

1. Shared utility functions for location-allocation (returned as callables):

```{python}
def prepare_location_model(df):
    df = df.rename(columns={"X": "x", "Y": "y", "Demand": "demand"}).copy()

    def assign_customers(df, facilities):
        assignments = []
        for _, row in df.iterrows():
            distances = [hypot(row['x'] - fx, row['y'] - fy) for fx, fy in facilities]
            assignments.append(np.argmin(distances))
        return np.array(assignments)

    def update_facilities(df, assignments, K):
        facilities = []
        for k in range(K):
            cluster = df[assignments == k]
            if not cluster.empty:
                total_demand = cluster['demand'].sum()
                x_center = (cluster['x'] * cluster['demand']).sum() / total_demand
                y_center = (cluster['y'] * cluster['demand']).sum() / total_demand
                facilities.append((x_center, y_center))
            else:
                facilities.append((0, 0))
        return facilities

    def compute_total_cost(df, facilities, assignments):
        total_cost = 0
        for i, (fx, fy) in enumerate(facilities):
            cluster = df[assignments == i]
            for _, row in cluster.iterrows():
                dist = hypot(row['x'] - fx, row['y'] - fy)
                total_cost += row['demand'] * dist
        return total_cost

    return assign_customers, update_facilities, compute_total_cost
```

2. Solve the facility location problem for a specific range of K:

```{python}
def solve_multi_facility_problem(df, K_values):
    assign_customers, update_facilities, compute_total_cost = prepare_location_model(df)
    df = df.rename(columns={"X": "x", "Y": "y", "Demand": "demand"}).copy()

    results = []
    for K in K_values:
        initial = df.sort_values('demand', ascending=False).head(K)
        facilities = list(zip(initial['x'], initial['y']))

        prev_assignments = None
        for _ in range(100):
            assignments = assign_customers(df, facilities)
            if np.array_equal(assignments, prev_assignments):
                break
            prev_assignments = assignments
            facilities = update_facilities(df, assignments, K)

        cost = compute_total_cost(df, facilities, assignments)
        results.append({"K": K, "Cost": cost})

    return pd.DataFrame(results)
```

3. Plot cost vs. number of warehouses

```{python}
def plot_elbow_curve(results_df):
    return alt.Chart(results_df).mark_line(point=True).encode(
        x=alt.X('K:O', title='Number of Warehouses (K)'),
        y=alt.Y('Cost', title='Total Distribution Cost'),
        tooltip=['K', 'Cost']
    ).properties(
        title='Elbow Method: Cost vs. Number of Warehouses'
    )
```

4. Run the algorithm for a specific K:

```{python}
def run_k_facilities(df, K):
    assign_customers, update_facilities, _ = prepare_location_model(df)
    df = df.rename(columns={"X": "x", "Y": "y", "Demand": "demand"}).copy()

    initial = df.sort_values('demand', ascending=False).head(K)
    facilities = list(zip(initial['x'], initial['y']))

    prev_assignments = None
    for _ in range(100):
        assignments = assign_customers(df, facilities)
        if np.array_equal(assignments, prev_assignments):
            break
        prev_assignments = assignments
        facilities = update_facilities(df, assignments, K)

    df['cluster'] = assignments
    facilities_df = pd.DataFrame(facilities, columns=['x', 'y'])
    facilities_df['cluster'] = [str(i) for i in range(K)]
    return df, facilities_df
```

5. Plot customer and warehouse locations

```{python}
def plot_facilities_and_customers(df_customers, df_facilities):
    df_customers['type'] = 'Customer'
    df_customers['size'] = df_customers['demand'] * 5
    df_customers['label'] = ''

    df_facilities['type'] = 'Warehouse'
    df_facilities['size'] = 500
    df_facilities['label'] = ['W' + str(i) for i in range(len(df_facilities))]

    combined = pd.concat([
        df_customers[['x', 'y', 'cluster', 'type', 'size', 'label']],
        df_facilities[['x', 'y', 'cluster', 'type', 'size', 'label']]
    ])

    combined['type'] = combined['type'].astype(str)  # ensure clean shape mapping

    points = (
        alt.Chart(combined)
        .mark_point(filled=True)
        .encode(
            x='x', y='y',
            color=alt.Color('cluster:N', title='Cluster'),
            shape=alt.Shape(
                'type:N', 
                scale=alt.Scale(
                    domain=['Customer', 'Warehouse'], 
                    range=['circle', 'triangle']
                )
            ),
            size='size',
            tooltip=['type', 'x', 'y', 'label']
        )
    )

    labels = alt.Chart(df_facilities).mark_text(
        align='left', dx=8, dy=-8, fontSize=13
    ).encode(
        x='x', y='y', text='label'
    )

    return (points + labels).properties(
        title='Customer and Warehouse Locations'
    )
```

##### Solving Steps

1. Load in the data:

```{python}
# The dataset contains customer coordinates (X, Y) and their demand.
df = pd.read_csv('data/multi_city_data.csv')  # Update path as needed
```

2. Solve the multi-facility problem and look at costs for different number of warehouses.

**Total Cost**: The total cost with multiple facilities is the sum of each customer’s demand * distance to its assigned facility. As we add facilities, customers have shorter distances, so total cost decreases. We can plot or tabulate the cost vs $K$:

```{python}
results = solve_multi_facility_problem(df, range(1, 11))
plot_elbow_curve(results)
```

From these results, we observe **diminishing marginal benefit** for additional facilities. Going from 1 to 2 warehouses yielded a large cost reduction; 2 to 3 also yielded a large reduction; but by 4 to 5, the incremental savings were smaller. This is typical: the first few facilities dramatically cut distance because they eliminate the longest transport lanes, but beyond a point, new facilities only shave off smaller remaining distances.

If we continued to $K=20$ (a warehouse for every customer), cost would drop to 0 because each customer would be at the same location as a warehouse (in theory). But of course, in reality, building 20 warehouses is impractical – and our model doesn’t include any fixed facility cost to penalize that.

3. Solve for facility locations & create a plot:

```{python}
df_k4, facilities_k4 = run_k_facilities(df, K=4)
```

```{python}
df_k4
```

```{python}
facilities_k4
```

```{python}
plot_facilities_and_customers(df_k4, facilities_k4)
```

Just for curiosity, we can also look at `k=3` to compare:

```{python}
df_k3, facilities_k3 = run_k_facilities(df, K=3)
plot_facilities_and_customers(df_k3, facilities_k3)
```

## Summary

In this module, we explored facility location models for supply chain network design:

- **Single-facility location**: We learned to formulate the problem of finding the optimal single facility location to minimize total weighted distance to customers. By using a squared-distance approximation, we could solve it with Gurobi as a quadratic program. The solution is essentially a weighted average of customer locations (center-of-gravity), which balances the travel cost to all customers. We also discussed how additional constraints (like limiting maximum distance) or different objectives (minimax vs min-sum) could affect the solution.
- **Multi-facility location**: We tackled the harder problem of placing multiple facilities. We introduced an iterative location-allocation heuristic (analogous to k-means clustering) to assign customers to the nearest facility and recompute facility positions until reaching a good solution. Through an example with 20 customers, we saw that adding more facilities reduces total transport cost but with diminishing returns. This method gave us insight into how customers naturally cluster around local "centers of gravity" when multiple warehouses are allowed.

**Practical implications**: Companies often use such models to decide how many warehouses or plants to operate and where to locate them. The models can become more complex if including factors like facility fixed costs, capacities, or discrete candidate locations (those aspects lead to MILP formulations, which we will explore later). However, the fundamental trade-off remains: more facilities mean higher fixed costs but lower transportation costs (and vice versa). The tools learned here allow analyzing the transportation cost side of that trade-off by optimally placing a given number of facilities.

## Exercises

### Center of Gravity Calculation (Single Facility)

Consider three customer locations with coordinates and demands: A at $(0,0)$ with demand 10, B at $(10,0)$ with demand 5, and C at $(5, 5)$ with demand 5:

1. Compute the weighted center of gravity of these points (assuming equal transport cost per unit distance).
2. If the facility is located at that center of gravity, what is the total distance-weighted cost?
3. Try shifting the facility a bit in any direction, would the total cost increase or decrease? (This is a conceptual check to confirm it's an optimum.)

::: {.callout-tip}
For 1: $x^* = \frac{100 + 510 + 55}{10+5+5}$, $y^ = \frac{100 + 50 + 55}{20}$. 
For 2: calculate $10d(A,facility)+5d(B,facility)+5d(C,facility)$. 
For 3: consider symmetry, the center-of-gravity should balance the pull of all points, so moving any direction increases distance to some heavily weighted point more than it decreases to another.
:::

::: {.callout title="Solution" collapse="true"}
**Customer Data:**

| Customer | Coordinates (x, y) | Demand |
|----------|--------------------|--------|
| A        | (0, 0)             | 10     |
| B        | (10, 0)            | 5      |
| C        | (5, 5)             | 5      |

**Weighted Center of Gravity:**

$$
x^* = \frac{10 \cdot 0 + 5 \cdot 10 + 5 \cdot 5}{20} = 3.75, \quad
y^* = \frac{10 \cdot 0 + 5 \cdot 0 + 5 \cdot 5}{20} = 1.25
$$

**Total Distance-Weighted Cost:**

$$
\text{Total Cost} = 10 \cdot d(A, \text{facility}) + 5 \cdot d(B, \text{facility}) + 5 \cdot d(C, \text{facility}) \approx \textbf{91.16}
$$
:::

### Multi-facility Assignment Intuition

Suppose you have four customers located at the four corners of a square region and all have equal demand. If you are to place $K=2$ facilities, where would you intuitively place them and which customers would each serve?

1. Draw a diagram of the four points and show two facility locations that make sense.
2. Explain why your assignment of customers to facilities is optimal in terms of distance.
3. How would the solution change if one of the four customers had a much higher demand than the others?

### Solving a Multi-Facility Problem

Use the new dataset of 15 customers provided in the sidebar (`Exercise Data`). Implement the iterative algorithm (or use the provided k-means helper functions) for different values of $K$:

1. Load the dataset and plot the customer locations.
2. For $K = 2, 3, 4$, run the algorithm and record:
  - The final total cost.
  - The coordinates of the facility locations.
3. Plot customer clusters and facility positions using Altair or Matplotlib.
4. Think about the trade-off you see between the number of warehouses and cost:
  - At what point do diminishing returns appear?
  - If the fixed cost of operating a warehouse is 500 units, how many warehouses should the company build?

::: {.callout title="Solution" collapse="true"}
**Results:**

| K | Total Cost | Cluster Centers (x, y) |
|---|------------|-------------------------|
| 2 | 1571.11    | (1.66, 3.79), (7.82, 6.62) |
| 3 | 1207.61    | (1.93, 1.65), (7.82, 6.62), (3.14, 9.55) |
| 4 | 799.36     | (1.93, 1.65), (6.94, 4.67), (3.14, 9.55), (8.70, 2.39) |

**Interpretation:**

- As the number of warehouses $K$ increases, the total cost decreases.
- The largest drop in cost occurs between $K = 2$ and $K = 3$.
- Diminishing returns start to appear around $K = 4$.
:::
