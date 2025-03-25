---
title: "06 | Minimum Cost Network Flow Models"
subtitle: "Optimizing Transportation in Supply Chain Networks"
format: 
  html:
    other-links:
      - text: "min_cost_ex_arcs.csv"
        href: "data/min_cost_ex_arcs.csv"
        icon: file-earmark-spreadsheet
      - text: "min_cost_ex_nodes.csv"
        href: "data/min_cost_ex_nodes.csv"
        icon: file-earmark-spreadsheet
      - text: "min_cost_ex_solution.csv"
        href: "data/min_cost_ex_solution.csv"
        icon: file-earmark-spreadsheet
      - text: "optimal_flows.csv (Oil Transport Problem)"
        href: "optimal_flows.csv"
        icon: file-earmark-spreadsheet
order: 6
---

## Overview

Network flow models are powerful tools for optimizing the movement of goods through a supply chain network. The minimum cost network flow model helps determine the most cost-effective way to transport products from origins (like factories) to destinations (like warehouses).

In this module, you will learn how to:

1. Understand the key components of network flow problems
2. Formulate these as linear programming models
3. Implement and solve them using both Excel Solver and Python/Gurobi
4. Analyze different scenarios to make better supply chain decisions

## Key Components of Network Flow Problems

A minimum cost network flow problem has three main components:

1. **Nodes**: Points in the network representing locations
   - **Supply nodes**: Where products originate (factories, plants)
   - **Transshipment nodes**: Intermediate points (distribution centers)
   - **Demand nodes**: Where products are required (warehouses, customers)

2. **Arcs**: Connections between nodes representing possible flow paths
   - Each arc has a **cost per unit** of flow
   - Each arc may have a **capacity limit**

3. **Flows**: The amount of resource moving along each arc
   - Flow must be conserved at each node (total inflow = total outflow, except at supply/demand nodes)
   - Total supply must equal total demand for the problem to have feasible solutions

## Mathematical Formulation

Let's define:

- $x_{ij}$ = flow from node $i$ to node $j$
- $c_{ij}$ = cost per unit flow from node $i$ to node $j$
- $u_{ij}$ = capacity of arc from node $i$ to node $j$
- $b_i$ = net flow at node $i$ (positive for supply, negative for demand, zero for transshipment)

The general formulation is:

**Objective**: Minimize $\sum_{(i,j)} c_{ij} \cdot x_{ij}$

**Subject to**:

1. Flow Conservation: $\sum_{j:(i,j)} x_{ij} - \sum_{j:(j,i)} x_{ji} = b_i$ for all nodes $i$
2. Capacity Constraints: $0 \leq x_{ij} \leq u_{ij}$ for all arcs $(i,j)$

## Example 01: Distribution Unlimited Co.

Let's apply this to a concrete example. Distribution Unlimited Co. has two factories (F1, F2) that need to ship products to two warehouses (W1, W2). There's also a distribution center (DC) that can be used as an intermediate point.

```{mermaid}
flowchart LR
    F1(Factory 1<br/>80 units)
    F2(Factory 2<br/>40 units)
    DC(Distribution Center)
    W1(Warehouse 1<br/>-60 units)
    W2(Warehouse 2<br/>-90 units)
    
    F1 -->|$7/unit| W1
    F1 -->|$3/unit<br/>Max 50 units| DC
    F2 -->|$4/unit<br/>Max 50 units| DC
    DC -->|$0/unit| W1
    DC -->|$2/unit<br/>Max 60 units| W2
```

The network includes:

- Factory 1 (F1) produces 80 units
- Factory 2 (F2) produces 40 units
- Warehouse 1 (W1) requires 60 units
- Warehouse 2 (W2) requires 90 units

Notice that total demand (150 units) exceeds total supply (120 units). This means we'll need to handle unfulfilled demand in our model.

### Handling Supply-Demand Imbalance

When total supply doesn't equal total demand, we can:

1. **Add a dummy supply node** (if supply < demand)
2. **Add a dummy demand node** (if supply > demand)

For Distribution Unlimited Co., we'll add a dummy supply of 30 units with zero transportation cost to make the problem feasible:

```{mermaid}
flowchart LR
    F1(Factory 1<br/>80 units)
    F2(Factory 2<br/>40 units)
    DC(Distribution Center)
    W1(Warehouse 1<br/>-60 units)
    W2(Warehouse 2<br/>-90 units)
    DS(Dummy Supply<br/>30 units)
    
    F1 -->|$7/unit| W1
    F1 -->|$3/unit<br/>Max 50 units| DC
    F2 -->|$4/unit<br/>Max 50 units| DC
    DC -->|$0/unit| W1
    DC -->|$2/unit<br/>Max 60 units| W2
    DS -->|$0/unit| W2
```

### Solving with Python

The Python implementation uses Gurobi Optimizer through its Python interface, `gurobipy`. Unlike Excel, which uses a tabular format, Python builds the model programmatically. Let's break down how this works step by step.

Here's the basic structure:

1. Define nodes and their supply/demand values
2. Define arcs with costs and capacities
3. Create decision variables for flows
4. Add flow conservation constraints
5. Add capacity constraints
6. Set the objective and solve the model

First, we need to import the necessary libraries:

```python
import gurobipy as gp
from gurobipy import GRB
```

Gurobipy is Gurobi's Python API, and GRB provides constants like GRB.MINIMIZE for setting the optimization direction.

#### Defining the Problem Structure

Next, we define our network structure using Python data structures:

```{python}
# Define nodes
nodes = ['F1', 'F2', 'DC', 'W1', 'W2', 'Dummy']

# Define supply/demand at each node
supply_demand = {
    'F1': 80,    # Supply of 80 units
    'F2': 40,    # Supply of 40 units
    'DC': 0,     # Transshipment node (no net supply/demand)
    'W1': -60,   # Demand of 60 units
    'W2': -90,   # Demand of 90 units
    'Dummy': 30  # Dummy supply of 30 units
}

# Define arcs with costs and capacities
arcs = {
    ('F1', 'W1'): (7, float('inf')),  # Direct route from F1 to W1
    ('F1', 'DC'): (3, 50),            # Route from F1 to DC with capacity 50
    ('F2', 'DC'): (4, 50),            # Route from F2 to DC with capacity 50
    ('DC', 'W1'): (0, float('inf')),  # Route from DC to W1
    ('DC', 'W2'): (2, 60),            # Route from DC to W2 with capacity 60
    ('Dummy', 'W2'): (0, float('inf'))  # Dummy route to W2
}
```

This approach is very flexible. We can easily add or remove nodes and arcs by modifying these data structures.

#### Creating the Optimization Model

Now we create a Gurobi model and define the flow variables:

```python
# Create a new model
model = gp.Model("MinCostNetworkFlow")

# Create flow variables for each arc
flow = {}
for (i, j), (cost, _) in arcs.items():
    flow[(i, j)] = model.addVar(name=f'flow_{i}_{j}', obj=cost)
```

Each variable `flow[(i, j)]` represents the amount flowing from node `i` to node `j`. The `obj=cost` parameter sets the coefficient in the objective function.

##### `flow = {}`

This initializes an empty Python dictionary called `flow`. This dictionary will store our decision variables, which represent the amount of product flowing along each arc in the network. The dictionary's keys will be tuples representing arcs (from node $i$ to node $j$), and the values will be Gurobi variable objects.

##### `arcs.items()`

The `arcs` variable is a dictionary where:

- Each key is a tuple `(i, j)` representing an arc from node $i$ to node $j$.
- Each value is another tuple `(cost, capacity)` containing the cost per unit of flow and the maximum capacity for that arc.

The `items()` method returns an iterator over the key-value pairs in the dictionary, so `arcs.items()` produces pairs like: `((i, j), (cost, capacity))`.

```{python}
for value in arcs.items():
    print(value)
```

##### `for (i, j), (cost, _) in arcs.items():`

This line uses Python's tuple unpacking to extract values. For each iteration:

- `(i, j)` captures the origin and destination nodes from the arc tuple.
- `(cost, _)` captures the cost while ignoring the capacity (the underscore `_` is a Python convention for a variable you don't intend to use).

For example, if one entry in `arcs` is `('F1', 'DC'): (3, 50)`, then in that iteration:

- `i` would be `'F1'` (the origin node)
- `j` would be `'DC'` (the destination node)
- `cost` would be `3` (the cost per unit of flow)
- `_` would be `50` (the capacity, which we're ignoring for now)

##### `flow[(i, j)] = model.addVar(name=f'flow_{i}_{j}', obj=cost)`

For each arc, this line creates a Gurobi decision variable and adds it to our flow dictionary:

- `model.addVar()` creates a new decision variable in the Gurobi model.
- `name=f'flow_{i}_{j}'` gives the variable a descriptive name like `"flow_F1_DC"`.
- `obj=cost` sets the variable's coefficient in the objective function to the cost of this arc.
- `flow[(i, j)] = ...` stores the created variable in our dictionary with the arc tuple as the key.

By default, Gurobi variables are non-negative (`lb = 0`) and continuous, which is what we want for flow variables in this network problem.

##### What's Happening Overall

This loop iterates through each possible shipping route (arc) in our network. For each arc, it:

1. Extracts the origin node, destination node, and shipping cost
2. Creates a variable representing "how much to ship from origin to destination"
3. Associates that variable with the arc's cost in the objective function
4. Stores the variable in a dictionary for later use when creating constraints

After this loop completes, the `flow` dictionary contains all the decision variables our model needs, with each variable properly connected to its cost in the objective function. This is the foundation of our optimization model, we're creating variables for each possible shipment route and telling Gurobi that we want to minimize the total shipping cost.

The next parts of the code will add constraints to ensure flow conservation (what comes in equals what goes out at each node) and to enforce capacity limits on certain routes.

#### Adding Flow Conservation Constraints

The most important constraints ensure flow conservation at each node:

```python
# Add flow conservation constraints for each node
for i in nodes:
    # Sum of all flows leaving node i
    outflow = gp.quicksum(flow[(i, j)] for (i2, j) in arcs.keys() if i2 == i)
    
    # Sum of all flows entering node i
    inflow  = gp.quicksum(flow[(j, i)] for (j, i2) in arcs.keys() if i2 == i)
    
    # Outflow - inflow = supply/demand
    model.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')
```

This loop creates one constraint for each node. The constraint ensures that:

- For supply nodes: $outflow - inflow = supply amount$
- For demand nodes: $outflow - inflow = -demand amount$
- For transshipment nodes: $outflow - inflow = 0$

##### `for i in nodes:`

This loop iterates through each node in our network. For every node (whether it's a factory, warehouse, or distribution center), we need to create a flow conservation constraint.

Flow conservation is a fundamental principle in network flow models. It states that for any node in the network:

- The amount of flow entering the node, minus
- The amount of flow leaving the node,
- Must equal the node's supply or demand value

This mirrors real-world physical constraints: products don't disappear or materialize within the network (except at supply or demand nodes).

##### `outflow = gp.quicksum(flow[(i, j)] for (i2, j) in arcs.keys() if i2 == i)`

This line calculates the total flow leaving node `i`:

1. `arcs.keys()` gives us all the arc tuples `(i2, j)` in our network.
2. `if i2 == i` filters for only those arcs where the origin node matches our current node `i`.
3. `flow[(i, j)]` retrieves the decision variable representing flow on the arc from node `i` to some other node `j`.
4. `gp.quicksum(...)` sums up all these flow variables.

Essentially, we're adding up all the flow variables for arcs that originate from node `i`.

The reason for using `i2` here is that we're unpacking tuples from `arcs.keys()`. Each tuple is an arc represented as `(origin, destination)`. We use `i2` as a temporary variable name to compare with our current node `i`.

##### `inflow = gp.quicksum(flow[(j, i)] for (j, i2) in arcs.keys() if i2 == i)`

Similarly, this line calculates the total flow entering node `i`:

1. `arcs.keys()` gives us all the arc tuples `(j, i2)` in our network.
2. `if i2 == i` filters for only those arcs where the destination node matches our current node `i`.
3. `flow[(j, i)]` retrieves the decision variable representing flow on the arc from some other node `j` to node `i`.
4. `gp.quicksum(...)` sums up all these flow variables.

This time, we're adding up all the flow variables for arcs that terminate at node `i`.

##### `model.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')`

Finally, we add the constraint to our model:

1. `outflow - inflow` calculates the net flow (outgoing minus incoming) at node `i`.
2. `supply_demand[i]` retrieves the supply/demand value for node `i`:
    - Positive supply for supply nodes (more going out than coming in)
    - Negative for demand nodes (more coming in than going out)
    - Zero for transshipment nodes (what comes in must equal what goes out)
3. `model.addConstr(...)` adds this constraint to the Gurobi model.
4. `name=f'node_{i}'` gives the constraint a unique name for identification.

The constraint enforces different behaviors depending on the node type:

- **Supply Nodes**: If `supply_demand[i] = 50`, then `outflow - inflow = 50`, meaning the node sends out 50 more units than it receives.
- **Demand Nodes**: If `supply_demand[i] = -30`, then `outflow - inflow = -30`, meaning the node receives 30 more units than it sends out.
- **Transshipment Nodes**: If `supply_demand[i] = 0`, then `outflow - inflow = 0`, meaning everything that enters the node must also leave it.

#### Adding Capacity Constraints

Some arcs have capacity limits:

```python
# Add capacity constraints for each arc
for (i, j), (_, capacity) in arcs.items():
    if capacity < float('inf'):
        model.addConstr(flow[(i, j)] <= capacity, name=f'capacity_{i}_{j}')
```

This only adds constraints for arcs with finite capacity. We use `float('inf')` to represent unlimited capacity.

##### `for (i, j), (_, capacity) in arcs.items():`

This loop iterates through each arc in our network and its corresponding capacity. Let's break down the unpacking:

1. `arcs.items()` returns pairs of `((i, j), (cost, capacity))` where:
    - `(i, j)` is the arc from node `i` to node `j`
    - `(cost, capacity)` contains the shipping cost and maximum capacity
2. `(i, j), (_, capacity)` unpacks these values:
    - `i` is the origin node
    - `j` is the destination node
    - `_` is the cost (we use underscore because we don't need the cost for capacity constraints)
    - `capacity` is the maximum amount that can flow on this arc

For example, if arcs contains ('F1', 'DC'): (3, 50), then in that iteration:

- `i` would be `'F1'` (origin)
- `j` would be `'DC'` (destination)
- `_` would be `3` (cost, ignored here)
- `capacity` would be `50` (maximum flow allowed)

##### `if capacity < float('inf'):`

Not all arcs need capacity constraints. Many routes have no practical limit, or their limit is so large that it won't affect the optimal solution. In our code:

- `float('inf')` represents infinity in Python
- `capacity < float('inf')` checks if the capacity is a finite number
    - If true, we need to add a capacity constraint
    - If false (capacity is infinite), no constraint is needed

This condition ensures we only add constraints where necessary, keeping our model streamlined.

##### `model.addConstr(flow[(i, j)] <= capacity, name=f'capacity_{i}_{j}')`

This line creates and adds the actual capacity constraint to our model:

1. `flow[(i, j)]` accesses the decision variable representing flow on this arc
2. `flow[(i, j)] <= capacity` creates a constraint saying the flow cannot exceed the capacity
3. `model.addConstr(...)` adds this constraint to the Gurobi model
4. `name=f'capacity_{i}_{j}'` gives the constraint a descriptive name like "capacity_F1_DC"

#### Setting the Objective and Solving

Finally, we set the objective function and solve the model:

```python
# Set objective to minimize total cost
model.ModelSense = GRB.MINIMIZE

# Optimize the model
model.optimize()
```

#### Extracting and Analyzing the Solution

After solving, we can extract and analyze the results:

```python
# Check if an optimal solution was found
if model.Status == GRB.OPTIMAL:
    # Extract the solution
    flow_values = {}
    total_cost = 0
    
    for (i, j), var in flow.items():
        # Get the flow amount for this arc
        flow_amount = var.X
        
        # Only include arcs with positive flow
        if flow_amount > 0.001:  # Small threshold for floating-point errors
            flow_values[(i, j)] = flow_amount
            
            # Add to the total cost
            cost = arcs[(i, j)][0]
            total_cost += flow_amount * cost
    
    print("Optimal flows:")
    for (i, j), flow in sorted(flow_values.items()):
        cost = arcs[(i, j)][0]
        print(f"{i} → {j}: {flow:.1f} units (cost: ${cost}/unit)")
    
    print(f"<br/>Total transportation cost: ${total_cost:.1f}")
else:
    print(f"No optimal solution found. Status code: {model.Status}")
```

This code extracts the flow values from the solution, calculates the total cost, and prints a summary of the optimal flows.

#### All Together

```{python}
import gurobipy as gp
from gurobipy import GRB

# Define nodes
nodes = ['F1', 'F2', 'DC', 'W1', 'W2', 'Dummy']

# Define supply/demand at each node
supply_demand = {
    'F1': 80,    # Supply of 80 units
    'F2': 40,    # Supply of 40 units
    'DC': 0,     # Transshipment node (no net supply/demand)
    'W1': -60,   # Demand of 60 units
    'W2': -90,   # Demand of 90 units
    'Dummy': 30  # Dummy supply of 30 units
}

# Define arcs with costs and capacities
arcs = {
    ('F1', 'W1'): (7, float('inf')),  # Direct route from F1 to W1
    ('F1', 'DC'): (3, 50),            # Route from F1 to DC with capacity 50
    ('F2', 'DC'): (4, 50),            # Route from F2 to DC with capacity 50
    ('DC', 'W1'): (0, float('inf')),  # Route from DC to W1
    ('DC', 'W2'): (2, 60),            # Route from DC to W2 with capacity 60
    ('Dummy', 'W2'): (0, float('inf'))  # Dummy route to W2
}

# Create a new model
model = gp.Model("MinCostNetworkFlow")
model.Params.LogToConsole = 0 # should turn off unwanted gurobipy output

# Create flow variables for each arc
flow = {}
for (i, j), (cost, _) in arcs.items():
    flow[(i, j)] = model.addVar(name=f'flow_{i}_{j}', obj=cost)

# Add flow conservation constraints for each node
for i in nodes:
    # Sum of all flows leaving node i
    outflow = gp.quicksum(flow[(i, j)] for (i2, j) in arcs.keys() if i2 == i)
    
    # Sum of all flows entering node i
    inflow = gp.quicksum(flow[(j, i)] for (j, i2) in arcs.keys() if i2 == i)
    
    # Outflow - inflow = supply/demand
    model.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')

# Add capacity constraints for each arc
for (i, j), (_, capacity) in arcs.items():
    if capacity < float('inf'):
        model.addConstr(flow[(i, j)] <= capacity, name=f'capacity_{i}_{j}')

# Set objective to minimize total cost
model.ModelSense = GRB.MINIMIZE

# Optimize the model
model.optimize()

# Check if an optimal solution was found
if model.Status == GRB.OPTIMAL:
    # Extract the solution
    flow_values = {}
    total_cost = 0
    
    for (i, j), var in flow.items():
        # Get the flow amount for this arc
        flow_amount = var.X
        
        # Only include arcs with positive flow
        if flow_amount > 0.001:  # Small threshold for floating-point errors
            flow_values[(i, j)] = flow_amount
            
            # Add to the total cost
            cost = arcs[(i, j)][0]
            total_cost += flow_amount * cost
    
    print("Optimal flows:")
    for (i, j), flow in sorted(flow_values.items()):
        cost = arcs[(i, j)][0]
        print(f"{i} → {j}: {flow:.1f} units (cost: ${cost}/unit)")
    
    print(f"<br/>Total transportation cost: ${total_cost:.1f}")
else:
    print(f"No optimal solution found. Status code: {model.Status}")
```

```{mermaid}
flowchart LR
    F1(Factory 1<br/>80 units)
    F2(Factory 2<br/>40 units)
    DC(Distribution Center)
    W1(Warehouse 1<br/>-60 units)
    W2(Warehouse 2<br/>-90 units)
    DS(Dummy Supply<br/>30 units)
    
    F1 -->|30 units<br/>$7/unit| W1
    F1 -->|50 units<br/>$3/unit| DC
    F2 -->|40 units<br/>$4/unit| DC
    DC -->|30 units<br/>$0/unit| W1
    DC -->|60 units<br/>$2/unit| W2
    DS -->|30 units<br/>$0/unit| W2
    
    style F1 fill:#d4f1c5
    style F2 fill:#d4f1c5
    style DS fill:#d4f1c5
    style W1 fill:#c5daf1
    style W2 fill:#c5daf1
    style DC fill:#f1e9c5
```

#### Advantages of the Python Approach

The Python/Gurobi approach offers several advantages:

- Scales to very large networks (hundreds of nodes and arcs)
- Easier to modify and analyze multiple scenarios
- Can be integrated with other Python tools for data analysis and visualization
- Faster for complex problems
- More sophisticated error handling and reporting

Here is a version of this problem with five factories, 3 distribution centers, and eight warehouse locations:

```{mermaid}
flowchart LR
    %% Supply Nodes (Factories)
    F1[Factory 1<br/>120 units]
    F2[Factory 2<br/>150 units]
    F3[Factory 3<br/>200 units]
    F4[Factory 4<br/>180 units]
    F5[Factory 5<br/>250 units]
    
    %% Transshipment Nodes (Distribution Centers)
    DC1[Distribution Center 1]
    DC2[Distribution Center 2]
    DC3[Distribution Center 3]
    
    %% Demand Nodes (Warehouses)
    W1[Warehouse 1<br/>-80 units]
    W2[Warehouse 2<br/>-110 units]
    W3[Warehouse 3<br/>-90 units]
    W4[Warehouse 4<br/>-130 units]
    W5[Warehouse 5<br/>-150 units]
    W6[Warehouse 6<br/>-100 units]
    W7[Warehouse 7<br/>-120 units]
    W8[Warehouse 8<br/>-120 units]
    
    %% Factory to DC connections
    F1 -->|$3/unit<br/>Max 100| DC1
    F1 -->|$4/unit<br/>Max 80| DC2
    F2 -->|$4/unit<br/>Max 90| DC1
    F2 -->|$3/unit<br/>Max 100| DC2
    F3 -->|$3/unit<br/>Max 120| DC2
    F3 -->|$2/unit<br/>Max 150| DC3
    F4 -->|$4/unit<br/>Max 100| DC2
    F4 -->|$3/unit<br/>Max 110| DC3
    F5 -->|$5/unit<br/>Max 80| DC1
    F5 -->|$2/unit<br/>Max 200| DC3
    
    %% Sample direct Factory to Warehouse connections
    F1 -.->|$8/unit| W1
    F2 -.->|$8/unit| W2
    F3 -.->|$9/unit| W4
    F4 -.->|$10/unit| W5
    F5 -.->|$8/unit| W8
    
    %% DC to Warehouse connections (showing representative examples)
    DC1 -->|$3/unit<br/>Max 100| W1
    DC1 -->|$4/unit<br/>Max 90| W2
    DC1 -->|$2/unit<br/>Max 120| W3
    DC2 -->|$3/unit<br/>Max 100| W3
    DC2 -->|$2/unit<br/>Max 150| W4
    DC2 -->|$4/unit<br/>Max 120| W5
    DC3 -->|$4/unit<br/>Max 90| W6
    DC3 -->|$3/unit<br/>Max 140| W7
    DC3 -->|$2/unit<br/>Max 120| W8
    
    %% Inter-DC connections
    DC1 -->|$1/unit<br/>Max 70| DC2
    DC2 -->|$1/unit<br/>Max 90| DC3
    DC3 -->|$2/unit<br/>Max 60| DC1
    
    %% Node styling
    classDef supply fill:#d4f1c5,stroke:#333,stroke-width:1px
    classDef transship fill:#f1e9c5,stroke:#333,stroke-width:1px
    classDef demand fill:#c5daf1,stroke:#333,stroke-width:1px
    
    %% Apply styles
    class F1,F2,F3,F4,F5 supply
    class DC1,DC2,DC3 transship
    class W1,W2,W3,W4,W5,W6,W7,W8 demand
```

```{python}
#| code-fold: true

"""
Minimum Cost Network Flow Solver

This script reads supply chain network data from CSV files and 
solves the minimum cost network flow problem using Gurobi optimizer.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Read the node data
print("Reading node data from nodes.csv...")
nodes_df = pd.read_csv(r'data\min_cost_ex_nodes.csv')

# Read the arc data
print("Reading arc data from arcs.csv...")
arcs_df = pd.read_csv(r'data\min_cost_ex_arcs.csv')

# Extract node information
nodes = nodes_df['node_id'].tolist()
supply_demand = dict(zip(nodes_df['node_id'], nodes_df['supply_demand']))
node_types = dict(zip(nodes_df['node_id'], nodes_df['type']))

# Group nodes by type for reporting
factories = [node for node, type_val in node_types.items() if type_val == 'Factory']
distribution_centers = [node for node, type_val in node_types.items() if type_val == 'DC']
warehouses = [node for node, type_val in node_types.items() if type_val == 'Warehouse']

# Check supply/demand balance
total_supply = sum(v for v in supply_demand.values() if v > 0)
total_demand = -sum(v for v in supply_demand.values() if v < 0)
print(f"Total supply: {total_supply}")
print(f"Total demand: {total_demand}")
if total_supply != total_demand:
    print("Warning: Supply and demand are not balanced!")

# Create arcs dictionary from DataFrame
# Convert 'inf' strings to actual infinity
arcs_df['capacity'] = arcs_df['capacity'].replace('inf', float('inf'))
arcs_df['capacity'] = pd.to_numeric(arcs_df['capacity'])

# Create arcs dictionary
arcs = {}
for _, row in arcs_df.iterrows():
    arcs[(row['from_node'], row['to_node'])] = (row['cost'], row['capacity'])

print(f"Network has {len(nodes)} nodes and {len(arcs)} arcs")

# Create a new Gurobi model
model = gp.Model("SupplyChainNetwork")
model.Params.LogToConsole = 0

# Create flow variables for each arc
flow = {}
for (i, j), (cost, _) in arcs.items():
    flow[(i, j)] = model.addVar(name=f'flow_{i}_{j}', obj=cost)

# Add flow conservation constraints for each node
for i in nodes:
    # Sum of flows leaving node i
    outflow = gp.quicksum(flow[(i, j)] for (i2, j) in arcs.keys() if i2 == i)
    
    # Sum of flows entering node i
    inflow = gp.quicksum(flow[(j, i)] for (j, i2) in arcs.keys() if i2 == i)
    
    # Outflow - inflow = supply/demand
    model.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')

# Add capacity constraints for each arc
for (i, j), (_, capacity) in arcs.items():
    if capacity < float('inf'):
        model.addConstr(flow[(i, j)] <= capacity, name=f'capacity_{i}_{j}')

# Set objective to minimize total cost
model.ModelSense = GRB.MINIMIZE

# Solve the model
print("\nSolving the network flow problem...")
model.optimize()

# Check if an optimal solution was found
if model.Status == GRB.OPTIMAL:
    # Calculate flow statistics
    active_arcs = 0
    capacity_limited_arcs = 0
    total_cost = 0
    
    # Dictionaries to track flows by node type
    factory_flows = {f: 0 for f in factories}
    dc_throughput = {dc: 0 for dc in distribution_centers}
    warehouse_inflows = {w: 0 for w in warehouses}
    
    # Extract solution
    flow_values = {}
    for (i, j), var in flow.items():
        flow_amount = var.X
        if flow_amount > 0.001:  # Only count non-zero flows
            flow_values[(i, j)] = flow_amount
            active_arcs += 1
            cost = arcs[(i, j)][0]
            capacity = arcs[(i, j)][1]
            total_cost += flow_amount * cost
            
            # Check if arc is at capacity
            if capacity < float('inf') and abs(flow_amount - capacity) < 0.001:
                capacity_limited_arcs += 1
            
            # Update node statistics
            if i in factories:
                factory_flows[i] += flow_amount
            if i in distribution_centers:
                dc_throughput[i] += flow_amount
            if j in warehouses:
                warehouse_inflows[j] += flow_amount
    
    # Print results
    print("\n========== OPTIMAL SOLUTION FOUND ==========")
    print(f"Total transportation cost: ${total_cost:.2f}")
    print(f"Active arcs: {active_arcs} out of {len(arcs)} possible")
    print(f"Arcs at capacity: {capacity_limited_arcs}")
    
    # Factory utilization
    print("\nFACTORY UTILIZATION:")
    for f in factories:
        utilization_pct = (factory_flows[f] / supply_demand[f]) * 100
        print(f"  {f}: {factory_flows[f]} units shipped ({utilization_pct:.1f}% of capacity)")
    
    # Distribution center throughput
    print("\nDISTRIBUTION CENTER THROUGHPUT:")
    for dc in distribution_centers:
        print(f"  {dc}: {dc_throughput[dc]} units processed")
    
    # Warehouse demand fulfillment
    print("\nWAREHOUSE DEMAND FULFILLMENT:")
    for w in warehouses:
        received_pct = (warehouse_inflows[w] / (-supply_demand[w])) * 100
        print(f"  {w}: {warehouse_inflows[w]} units received ({received_pct:.1f}% of demand)")
    
    # Detailed flow report
    print("\nDETAILED FLOW REPORT (non-zero flows only):")
    
    # Factory to warehouse direct
    f_to_w = [(i, j, flow_values[(i, j)]) for (i, j) in flow_values.keys() 
             if i in factories and j in warehouses]
    if f_to_w:
        print("\n  Factory → Warehouse (Direct):")
        for i, j, amt in sorted(f_to_w):
            cost = arcs[(i, j)][0]
            print(f"    {i} → {j}: {amt:.1f} units (cost: ${cost}/unit, total: ${amt*cost:.1f})")
    
    # Factory to DC
    f_to_dc = [(i, j, flow_values[(i, j)]) for (i, j) in flow_values.keys() 
              if i in factories and j in distribution_centers]
    if f_to_dc:
        print("\n  Factory → Distribution Center:")
        for i, j, amt in sorted(f_to_dc):
            cost = arcs[(i, j)][0]
            capacity = arcs[(i, j)][1]
            at_capacity = " (at capacity)" if abs(amt - capacity) < 0.001 else ""
            print(f"    {i} → {j}: {amt:.1f} units (cost: ${cost}/unit, total: ${amt*cost:.1f}){at_capacity}")
    
    # DC to warehouse
    dc_to_w = [(i, j, flow_values[(i, j)]) for (i, j) in flow_values.keys() 
              if i in distribution_centers and j in warehouses]
    if dc_to_w:
        print("\n  Distribution Center → Warehouse:")
        for i, j, amt in sorted(dc_to_w):
            cost = arcs[(i, j)][0]
            capacity = arcs[(i, j)][1]
            at_capacity = " (at capacity)" if abs(amt - capacity) < 0.001 else ""
            print(f"    {i} → {j}: {amt:.1f} units (cost: ${cost}/unit, total: ${amt*cost:.1f}){at_capacity}")
    
    # Inter-DC flows
    dc_to_dc = [(i, j, flow_values[(i, j)]) for (i, j) in flow_values.keys() 
               if i in distribution_centers and j in distribution_centers]
    if dc_to_dc:
        print("\n  Distribution Center → Distribution Center:")
        for i, j, amt in sorted(dc_to_dc):
            cost = arcs[(i, j)][0]
            capacity = arcs[(i, j)][1]
            at_capacity = " (at capacity)" if abs(amt - capacity) < 0.001 else ""
            print(f"    {i} → {j}: {amt:.1f} units (cost: ${cost}/unit, total: ${amt*cost:.1f}){at_capacity}")
    
    # Export solution to CSV
    solution_data = []
    for (i, j), flow_amount in flow_values.items():
        cost = arcs[(i, j)][0]
        capacity = arcs[(i, j)][1]
        solution_data.append({
            'from_node': i,
            'to_node': j,
            'flow': flow_amount,
            'cost_per_unit': cost,
            'total_cost': flow_amount * cost,
            'capacity': capacity if capacity < float('inf') else 'unlimited',
            'at_capacity': 'Yes' if capacity < float('inf') and abs(flow_amount - capacity) < 0.001 else 'No'
        })
    
    solution_df = pd.DataFrame(solution_data)
    solution_df.to_csv('solution.csv', index=False)
    print()
    print(solution_df)
    
else:
    print(f"No optimal solution found. Status code: {model.Status}")
    print("Check your data for inconsistencies in supply/demand balance or network connectivity.")
```

This is just for example, we will not being doing any problems like this in this course.

## Example 02: Oil Transport Problem

Let's look at another example. The Conch Oil Company needs to transport 30 million barrels of crude oil from Doha, Qatar to three European refineries:

- Rotterdam, Netherlands (6 million barrels)
- Toulon, France (15 million barrels)
- Palermo, Italy (9 million barrels)

There are three possible routes:

1. Direct shipping around Africa (most expensive)
2. Through the Suez Canal to Port Said, then to destinations
3. From Suez to Damietta via pipeline (limited to 15 million barrels), then to destinations

```{mermaid}
flowchart LR
    Doha(Doha<br/>30M barrels)
    Suez(Suez)
    PortSaid(Port Said)
    Damietta(Damietta)
    Rotterdam(Rotterdam<br/>-6M barrels)
    Toulon(Toulon<br/>-15M barrels)
    Palermo(Palermo<br/>-9M barrels)
    
    Doha -->|$1.20/barrel| Rotterdam
    Doha -->|$1.40/barrel| Toulon
    Doha -->|$1.35/barrel| Palermo
    Doha -->|$0.35/barrel| Suez
    Suez -->|$0.20/barrel| PortSaid
    Suez -->|$0.16/barrel<br/>Max 15M barrels| Damietta
    PortSaid -->|$0.27/barrel| Rotterdam
    PortSaid -->|$0.28/barrel| Toulon
    PortSaid -->|$0.19/barrel| Palermo
    Damietta -->|$0.25/barrel| Rotterdam
    Damietta -->|$0.20/barrel| Toulon
    Damietta -->|$0.15/barrel| Palermo
    
    style Doha fill:#d4f1c5
    style Rotterdam fill:#c5daf1
    style Toulon fill:#c5daf1
    style Palermo fill:#c5daf1
    style Suez fill:#f1e9c5
    style PortSaid fill:#f1e9c5
    style Damietta fill:#f1e9c5
```

### Optimal Solution to Oil Transport Problem

Let's implement a complete Python solution for the Conch Oil Company problem. We'll structure this similar to our previous example but add more detailed code comments to explain the logic.

```{python}
import gurobipy as gp
from gurobipy import GRB

# Define the nodes in our network
nodes = ['Doha', 'Suez', 'PortSaid', 'Damietta', 'Rotterdam', 'Toulon', 'Palermo']

# Define supply/demand at each node (in millions of barrels)
supply_demand = {
    'Doha': 30,        # Supply: 30M barrels at origin
    'Suez': 0,         # Transshipment node: no net supply/demand
    'PortSaid': 0,     # Transshipment node: no net supply/demand
    'Damietta': 0,     # Transshipment node: no net supply/demand
    'Rotterdam': -6,   # Demand: 6M barrels
    'Toulon': -15,     # Demand: 15M barrels
    'Palermo': -9      # Demand: 9M barrels
}

# Define arcs with (cost, capacity) tuples
# Format: (origin, destination): (cost per barrel, capacity in millions of barrels)
arcs = {
    # Direct routes from Doha to refineries
    ('Doha', 'Rotterdam'): (1.20, float('inf')),
    ('Doha', 'Toulon'): (1.40, float('inf')),
    ('Doha', 'Palermo'): (1.35, float('inf')),
    
    # Route via Suez
    ('Doha', 'Suez'): (0.35, float('inf')),
    ('Suez', 'PortSaid'): (0.20, float('inf')),
    ('Suez', 'Damietta'): (0.16, 15),  # Pipeline has 15M barrel capacity
    
    # Routes from Port Said to refineries
    ('PortSaid', 'Rotterdam'): (0.27, float('inf')),
    ('PortSaid', 'Toulon'): (0.28, float('inf')),
    ('PortSaid', 'Palermo'): (0.19, float('inf')),
    
    # Routes from Damietta to refineries
    ('Damietta', 'Rotterdam'): (0.25, float('inf')),
    ('Damietta', 'Toulon'): (0.20, float('inf')),
    ('Damietta', 'Palermo'): (0.15, float('inf'))
}

# Create a new Gurobi model
model = gp.Model("OilTransportProblem")
model.Params.LogToConsole = 0

# Create decision variables for flow on each arc
# Each variable represents millions of barrels flowing on that route
flow = {}
for (i, j), (cost, _) in arcs.items():
    # obj=cost sets this variable's coefficient in the objective function
    flow[(i, j)] = model.addVar(name=f'flow_{i}_{j}', obj=cost)

# Add flow conservation constraints
# For each node: outflow - inflow = supply/demand
for i in nodes:
    # Calculate total outflow from node i
    outflow = gp.quicksum(flow[(i, j)] for (i2, j) in arcs.keys() if i2 == i)
    
    # Calculate total inflow to node i
    inflow = gp.quicksum(flow[(j, i)] for (j, i2) in arcs.keys() if i2 == i)
    
    # Set constraint: outflow - inflow = supply/demand for this node
    model.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')

# Add capacity constraints for arcs with limited capacity
for (i, j), (_, capacity) in arcs.items():
    if capacity < float('inf'):
        model.addConstr(flow[(i, j)] <= capacity, name=f'capacity_{i}_{j}')

# Set objective to minimize total cost
model.ModelSense = GRB.MINIMIZE

# Solve the model
model.optimize()
```

When executing this model, the solution shows that:

1. All 30 million barrels are shipped from Doha to Suez
2. From Suez, 15 million barrels go through the pipeline to Damietta (utilizing full capacity)
3. The remaining 15 million barrels go to Port Said
4. From Damietta and Port Said, the oil is distributed to the refineries through the most cost-effective routes

We can extract values from our model with the following code. You should be able to use the same code for similar models, like in the lab (hint hint):

```{python}
import pandas as pd

# Create a list to store flow data for the CSV
flow_data = []
total_cost = 0

# Extract non-zero flows
for (i, j), var in flow.items():
    flow_amount = var.X
    
    # Filter out very small flows (numerical precision issues)
    if flow_amount > 0.001:
        # Get the cost for this arc
        cost = arcs[(i, j)][0]
        
        # Calculate the cost contribution
        flow_cost = flow_amount * cost
        total_cost += flow_cost
        
        # Add a row with all relevant information
        flow_data.append({
            'origin': i,
            'destination': j,
            'flow': round(flow_amount, 2),
            'cost_per_unit': cost,
            'total_cost': round(flow_cost, 2),
            'capacity': arcs[(i, j)][1] if arcs[(i, j)][1] < float('inf') else 'Unlimited'
        })

# Check if we have any flow data before proceeding
if flow_data:
    # Create a DataFrame from the flow data
    flow_df = pd.DataFrame(flow_data)
    
    # Sort the DataFrame without assuming column names
    if 'origin' in flow_df.columns and 'destination' in flow_df.columns:
        flow_df = flow_df.sort_values(by=['origin', 'destination'])
    
    # Save to CSV
    flow_df.to_csv('optimal_flows.csv', index=False)
    
    # Create a summary DataFrame with additional statistics
    summary_data = [{
        'metric': 'Total Cost',
        'value': round(total_cost, 2)
    }, {
        'metric': 'Total Flow Units',
        'value': round(sum(item['flow'] for item in flow_data), 2)
    }, {
        'metric': 'Number of Active Routes',
        'value': len(flow_data)
    }]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('solution_summary.csv', index=False)
    
    print(f"Solution exported to 'optimal_flows.csv' and 'solution_summary.csv'")
    print(f"Total transportation cost: ${total_cost:.2f}")
    print()
    print(summary_df)
```

This code:

1. Creates a structured dataset with all key information for each flow
2. Exports the detailed flows to 'optimal_flows.csv'
3. Creates a separate summary file with key metrics
4. Works with any network structure (not specific to the oil transport problem)
5. Preserves the total cost calculation from the original code

You can easily adapt this by:

1. Changing the column names if needed
2. Adding more metrics to the summary file
3. Modifying the rounding precision
4. Adding more details to each flow record

The CSV output will have a clean, tabular structure that can be opened in Excel or other tools for further analysis or visualization.

## References and Resources

1. Hillier & Lieberman, "Introduction to Operations Research," Chapter 9
2. The [Gurobi Modeling Examples](https://gurobi.github.io/modeling-examples/) repository

## Exercises

### Simple Network Flow Problem

A shipping company needs to transport goods from two origins (O1, O2) to three destinations (D1, D2, D3). The shipping costs (in $ per unit) and supply/demand quantities are shown below:

**Supply and Demand**:

- Origin O1 has 150 units available
- Origin O2 has 250 units available
- Destination D1 requires 100 units
- Destination D2 requires 200 units
- Destination D3 requires 100 units

**Shipping Costs**:

| From | To | Cost per unit |
|:----:|:--:|:-------------:|
| O1   | D1 | $5            |
| O1   | D2 | $3            |
| O1   | D3 | $6            |
| O2   | D1 | $4            |
| O2   | D2 | $6            |
| O2   | D3 | $2            |

Formulate and solve this minimum cost network flow problem.

#### Solution

The solution demonstrates how to model and solve a basic transportation problem using Gurobi. The key components include:

1. Setting up the network structure with origins (O1, O2) and destinations (D1, D2, D3)
2. Defining supply and demand quantities at each node
3. Creating shipping costs between each origin-destination pair
4. Formulating flow conservation constraints for all nodes
5. Solving the model to find the minimum cost solution

```{python}
#| code-fold: true

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Define the nodes in our network
origins = ['O1', 'O2']
destinations = ['D1', 'D2', 'D3']
nodes = origins + destinations

# Define supply/demand at each node
supply_demand = {
    'O1':  150,  # Origin 1 supplies 150 units
    'O2':  250,  # Origin 2 supplies 250 units
    'D1': -100,  # Destination 1 demands 100 units
    'D2': -200,  # Destination 2 demands 200 units
    'D3': -100   # Destination 3 demands 100 units
}

# Define shipping costs
shipping_costs = {
    ('O1', 'D1'): 5,
    ('O1', 'D2'): 3,
    ('O1', 'D3'): 6,
    ('O2', 'D1'): 4,
    ('O2', 'D2'): 6,
    ('O2', 'D3'): 2
}

# Define arcs with (cost, capacity)
arcs = {}
for (origin, dest), cost in shipping_costs.items():
    arcs[(origin, dest)] = (cost, float('inf'))  # All routes have unlimited capacity initially

# Create a new Gurobi model
model = gp.Model("SimpleTransportationProblem")
model.Params.LogToConsole = 0

# Create flow variables for each arc
flow = {}
for (i, j), (cost, _) in arcs.items():
    flow[(i, j)] = model.addVar(name=f'flow_{i}_{j}', obj=cost)

# Add flow conservation constraints
for i in nodes:
    # Sum of flows leaving node i
    outflow = gp.quicksum(flow[(i, j)] for (i2, j) in arcs.keys() if i2 == i)
    
    # Sum of flows entering node i
    inflow = gp.quicksum(flow[(j, i)] for (j, i2) in arcs.keys() if i2 == i)
    
    # Outflow - inflow = supply/demand
    model.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')

# Set objective to minimize total cost
model.ModelSense = GRB.MINIMIZE

# Solve the model
model.optimize()

# Extract the solution
flow_values = {}
total_cost = 0

for (i, j), var in flow.items():
    flow_amount = var.X
    
    # Only include arcs with positive flow
    if flow_amount > 0.001:
        flow_values[(i, j)] = flow_amount
        
        # Add to the total cost
        cost = arcs[(i, j)][0]
        total_cost += flow_amount * cost

print("\nOptimal shipping plan:")
for (origin, dest), amount in sorted(flow_values.items()):
    cost = arcs[(origin, dest)][0]
    print(f"{origin} → {dest}: {amount:.1f} units (cost: ${cost}/unit, total: ${amount*cost:.1f})")

print(f"\nTotal transportation cost: ${total_cost:.1f}")

# Export results to CSV (as requested)
flow_data = []
for (i, j), amount in flow_values.items():
    cost = arcs[(i, j)][0]
    flow_data.append({
        'origin': i,
        'destination': j,
        'flow': round(amount, 1),
        'cost_per_unit': cost,
        'total_cost': round(amount * cost, 1)
    })

flow_df = pd.DataFrame(flow_data)
flow_df.to_csv('exercise1_solution.csv', index=False)
print()
print(flow_df)
```

The optimal solution shows how to allocate shipments to minimize the total transportation cost. In the optimal solution, we typically see that cheaper routes are preferred (like O1→D2 with cost $3/unit) over more expensive alternatives. The CSV export functionality demonstrates how to save results for further analysis.

If you've noticed by now, a lot of the Python code is cookie-cutter once you create the initial setup.

### Network with Capacity Constraints

Extend Exercise 1 by adding the following capacity constraints:

- The route from O1 to D2 can handle at most 80 units
- The route from O2 to D3 can handle at most 60 units

Answer the following:

1. How do these constraints change the optimal solution?
2. Which routes are now at capacity?
3. How much does the total cost increase due to these constraints?

#### Solution

This solution extends Exercise 1 by adding capacity constraints on specific routes:

- O1 to D2: maximum 80 units
- O2 to D3: maximum 60 units

The solution approach:

1. Solves both the unconstrained and constrained versions of the problem
2. Compares the solutions to understand the impact of capacity constraints
3. Identifies which routes are at capacity in the optimal solution
4. Calculates the cost increase due to the constraints

```{python}
#| code-fold: true

# Define capacity constraints
capacity_constraints = {
    ('O1', 'D2'): 80,   # Route from O1 to D2 has max capacity of 80 units
    ('O2', 'D3'): 60    # Route from O2 to D3 has max capacity of 60 units
}

# Update arcs with capacity constraints
for (origin, dest) in capacity_constraints:
    cost = arcs[(origin, dest)][0]
    arcs[(origin, dest)] = (cost, capacity_constraints[(origin, dest)])

    # Create a new Gurobi model
model_constrained = gp.Model("CapacitatedTransportationProblem")
model_constrained.Params.LogToConsole = 0

# Create flow variables for each arc
flow_constrained = {}
for (i, j), (cost, _) in arcs.items():
    flow_constrained[(i, j)] = model_constrained.addVar(name=f'flow_{i}_{j}', obj=cost)

# Add flow conservation constraints
for i in nodes:
    # Sum of flows leaving node i
    outflow = gp.quicksum(flow_constrained[(i, j)] for (i2, j) in arcs.keys() if i2 == i)
    
    # Sum of flows entering node i
    inflow = gp.quicksum(flow_constrained[(j, i)] for (j, i2) in arcs.keys() if i2 == i)
    
    # Outflow - inflow = supply/demand
    model_constrained.addConstr(outflow - inflow == supply_demand[i], name=f'node_{i}')

# Add capacity constraints
for (i, j), (_, capacity) in arcs.items():
    if capacity < float('inf'):
        model_constrained.addConstr(flow_constrained[(i, j)] <= capacity, name=f'capacity_{i}_{j}')

# Set objective to minimize total cost
model_constrained.ModelSense = GRB.MINIMIZE

# Solve the model
model_constrained.optimize()

# Extract the solution
flow_values_constrained = {}
total_cost_constrained = 0

for (i, j), var in flow_constrained.items():
    flow_amount = var.X
    
    # Only include arcs with positive flow
    if flow_amount > 0.001:
        flow_values_constrained[(i, j)] = flow_amount
        
        # Add to the total cost
        cost = arcs[(i, j)][0]
        total_cost_constrained += flow_amount * cost

print("\nConstrained Solution:")
print("-" * 30)

for (origin, dest), amount in sorted(flow_values_constrained.items()):
    cost = arcs[(origin, dest)][0]
    capacity = arcs[(origin, dest)][1]
    
    capacity_info = ""
    if capacity < float('inf') and abs(amount - capacity) < 0.1:
        capacity_info = " (AT CAPACITY)"
    
    print(f"{origin} → {dest}: {amount:.1f} units (cost: ${cost}/unit, total: ${amount*cost:.1f}){capacity_info}")

print(f"\nTotal transportation cost: ${total_cost_constrained:.1f}")

# Calculate cost difference
cost_difference = total_cost_constrained - total_cost # from ex 1 solution
percentage_increase = (cost_difference / total_cost) * 100

print("\nCost Comparison:")
print(f"Cost increase due to capacity constraints: ${cost_difference:.1f} ({percentage_increase:.1f}%)")

# Export results to CSV
flow_data = []

# Add unconstrained solution data
for (i, j), amount in flow_values.items():
    cost = arcs[(i, j)][0]
    capacity = arcs[(i, j)][1]
    at_capacity = "Yes" if capacity < float('inf') and abs(amount - capacity) < 0.1 else "No"
    
    flow_data.append({
        'scenario': "Unconstrained",
        'origin': i,
        'destination': j,
        'flow': round(amount, 1),
        'cost_per_unit': cost,
        'total_cost': round(amount * cost, 1),
        'capacity': capacity if capacity < float('inf') else "Unlimited",
        'at_capacity': at_capacity
    })

# Add constrained solution data
for (i, j), amount in flow_values_constrained.items():
    cost = arcs[(i, j)][0]
    capacity = arcs[(i, j)][1]
    at_capacity = "Yes" if capacity < float('inf') and abs(amount - capacity) < 0.1 else "No"
    
    flow_data.append({
        'scenario': "Constrained",
        'origin': i,
        'destination': j,
        'flow': round(amount, 1),
        'cost_per_unit': cost,
        'total_cost': round(amount * cost, 1),
        'capacity': capacity if capacity < float('inf') else "Unlimited",
        'at_capacity': at_capacity
    })

flow_df = pd.DataFrame(flow_data)
print(flow_df)
```

The comparison provides an important real-world insight: limited shipping capacity often leads to higher transportation costs as companies are forced to use less efficient routes. The solution exports detailed results to a CSV file for further analysis, showing both the unconstrained and constrained flows side by side.

This exercise demonstrates how to implement and analyze capacity constraints in transportation networks, which is a common challenge in real-world supply chain management.
