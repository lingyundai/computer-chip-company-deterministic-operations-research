#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
from gurobipy import *
import numpy as np

# read excel sheets with pandas
data = pd.ExcelFile('SuperChipData.xlsx')
productionCapacityDf = pd.read_excel(data, 'Production Capacity')
salesRegionDemandDf = pd.read_excel(data, 'Sales Region Demand')
shippingCostsDf = pd.read_excel(data, 'Shipping Costs')
productionCostsDf = pd.read_excel(data, 'Production Costs')

# preparing for adding variables
# total number of sales region j
salesRegion = 23
# total number of computer chip z
computerChip = 30
# [sales region, computer chip, yearly demand(thousands)]
salesRegionDemand = salesRegionDemandDf.to_numpy()
# [facility name, facility yearly production capacity(thousands)]
productionCapacity = productionCapacityDf.to_numpy()
# [facility, computer chip, production cost per chip ($)]
productionCosts = productionCostsDf.to_numpy()
# [facility, computer chip, sales region, shipping cost per chip($)]
shippingCosts = shippingCostsDf.to_numpy()

# initialize model
m = Model()
# set objective function to be minimization
m.modelSense = GRB.MINIMIZE

# declare variables and add them to model
# facility set S
facilitySet = ["Alexandria", "Richmond", "Norfolk", "Roanoke", "Charolottesville"]
# assign number to facility
facilityDict = {}
for i, f in enumerate(facilitySet):
     facilityDict[f] = i + 1

# Djz = z chip yearly demand in sales region j (thousands unit)
D = {}
for d in salesRegionDemand:
        # (j, z) = demand in j z
        D[d[0], d[1]] = d[2]

# Ci = yearly chip production capacity in facility i (thousands unit)
C = {}
for c in productionCapacity:
     # i = production capacity in i
     C[facilityDict[c[0]]] = c[1]

# Piz = production cost per z chip in facility i ($)
P = {}
for p in productionCosts:
     # (i, z) = production cost in i, z
     P[facilityDict[p[0]], p[1]] = p[2]

# Sijz = shipping cost per z chip from facility i to sales region j ($)
S = {}
for s in shippingCosts:
     # (i, j, z) = shipping cost per z from i to j
     S[facilityDict[s[0]], s[2], s[1]] = s[3]

# Xijz = amount of z chips shipped from facility i to sales region j (unit)
X = {}
# iterate through facility set, sales region numbers and computer chip numbers
for i in facilitySet:
    for j in range(1, salesRegion + 1):
        for z in range(1, computerChip + 1):
            X[facilityDict[i], j, z] = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                                name = f"X_{i}_{j}_{z}")

# Yiz = amount of z chips produced in facility i (unit)
Y = {}
# iterate through facility set, computer chip numbers
for i in facilitySet:
    for z in range(1, computerChip + 1):
        Y[facilityDict[i], z] = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                         name = f"Y_{i}_{z}")
        
# notify model the changes
m.update()

# set the objective function
m.setObjective(quicksum(S[facilityDict[i], j, z] * X[facilityDict[i], j, z] for i 
                        in facilitySet for j in range(1, salesRegion + 1) for z in 
                        range(1, computerChip + 1)) + 
               quicksum(P[facilityDict[i], z] * Y[facilityDict[i], z] for i 
                        in facilitySet for z in range(1, computerChip + 1)))

# first constraint ∑Yiz <= 1000Ci
facilityProductionConst = {}
for i in facilitySet:
     facilityProductionConst[i] = m.addConstr(quicksum(
          Y[facilityDict[i], z] for z in range(1, computerChip + 1)
          ) <= 1000 * C[facilityDict[i]], name = f"production_{i}")

# second constraint ∑Xijz = 1000Djz
shipDemandConst = {}
for j in range(1, salesRegion + 1):
     for z in range(1, computerChip + 1):
          shipDemandConst[j, z] = m.addConstr(quicksum(
               X[facilityDict[i], j, z] for i in facilitySet) == 1000 * D[j, z],
               name = f"shipping_demand_{j}_{z}")

# third constraint Yiz >= ∑Xijz
productionShipConst = {}
for i in facilitySet:
     for z in range(1, computerChip + 1):
          productionShipConst[facilityDict[i], z] = m.addConstr(
               Y[facilityDict[i], z] >= quicksum(
                    X[facilityDict[i], j, z] for j in range(1, salesRegion + 1)), 
                    name = f"production_shipping_{i}_{z}")

# for the fourth and fifth constraints, since Xijz and Yiz both have lower bound of
# 0 on initialization, not adding them here

# notify model the changes
m.update()

# trigger optimization
m.optimize()

# print the object function value
print ("Optimal Production and Distribution Cost = ", m.objVal)
# write model to file
m.write("Computer-Chip-Project-Model.lp")
# write solution to file
m.write("Computer-Chip-Project-Model.sol")



# In[2]:


# To answer question 1 - 

# find out the total production capacity in all facilities
# initialize
totalProductionCapacity = 0
# add it up
for c in productionCapacity:
    totalProductionCapacity += c[1]
# in thousand unit
totalProductionCapacity *= 1000

# lookup map to store facility x has 
# y% of the total production capacity 
facilityPercentageDict = {}
for f, p in productionCapacity:
    facilityPercentageDict[f] = (p * 1000) / totalProductionCapacity
    print(f"Facility {f} Production Percentage: {facilityPercentageDict[f]}%")

# every chip's total demand
chipDemandDict = {}
for i, j, z in salesRegionDemand:
    if j not in chipDemandDict:
        chipDemandDict[j] = 0
    else:
        chipDemandDict[j] += (1000 * z)

# the current policy cost -
currentPolicyProductionCost = 0
# amount of chips produced in facility on y%
chipsProducedInFacilityDict = {}
for i in facilitySet:
    for z in range(1, computerChip + 1):  
        # facility i produces y% of every chip's total demand
        chipsProducedInFacilityDict[(i, z)] = chipDemandDict[z] * facilityPercentageDict[i]
print("Current Amount of Chips Produced in Facility (unit): ", chipsProducedInFacilityDict)
# calculate total policy cost based on facility chip production cost
for i, z, c in productionCosts:
    if (i, z) in chipsProducedInFacilityDict:
        currentPolicyProductionCost += chipsProducedInFacilityDict[(i, z)] * c
print("Current Policy Total Production Costs ($): ", currentPolicyProductionCost)

# calculate alternative policy production cost
newPolicyProductionCost = 0
for i, z, c in productionCosts:
    if (facilityDict[i], z) in Y:
        newPolicyProductionCost += Y[facilityDict[i], z].x * c
print("New Policy Total Production Costs ($): ", newPolicyProductionCost)
# positive means alternative policy is suboptimal, negative means more optimal
print("Policy Production Costs Difference ($): ", 
    newPolicyProductionCost - currentPolicyProductionCost)

# find out the new policy production percentage based
# on optimal objective function value -
facilityChipProductionDict = {}
for i in facilitySet:
    for z in range(1, computerChip + 1):
        # find out current percentage based on chip's total demand
        facilityChipProductionDict[(i, z)] = Y[facilityDict[i], z].x / chipDemandDict[z]
print("Optimal Policy Production Percentage: ", facilityChipProductionDict)


# In[3]:


# To answer question 2 - 

# function to calculate total shipping cost per facility
def total_shipping_cost_per_facility(costDict, shippingCosts, X):
    for i, z, j, c in shippingCosts:
        if i not in costDict:
            costDict[i] = 0
        else:
            costDict[i] += (X[facilityDict[i], j, z].x * c)
    return costDict

# function to calculate total production cost per facility
def total_production_cost_per_facility(costDict, productionCosts, Y):
    for i, z, c in productionCosts:
        if i not in costDict:
            costDict[i] = 0
        else:
            costDict[i] += (Y[facilityDict[i], z].x * c)
    return costDict

# helper function to increase facility capacity one by one for analyses
def capacity_increase_helper(increaseAmount, capacityArr, i, j):
    newCapacity = np.copy(capacityArr)
    newCapacity[i][j] += increaseAmount
    return newCapacity

# generate new Ci = yearly chip production capacity in facility i (thousands unit)
# after the capacity increase
def chip_production_per_facility(newCapacityArr, capacityDict):
    for c in newCapacityArr:
         # i = production capacity in i
         capacityDict[facilityDict[c[0]]] = c[1]
    return capacityDict

# generate model with respective new facility capacity
def model_builder(m, X, Y, capacityDict, facilityProductionConst, shipDemandConst, productionShipConst):
    # set objective function to be minimization
    m.modelSense = GRB.MINIMIZE

    # iterate through facility set, sales region numbers and computer chip numbers
    for i in facilitySet:
        for j in range(1, salesRegion + 1):
            for z in range(1, computerChip + 1):
                X[facilityDict[i], j, z] = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                                    name = f"X_{i}_{j}_{z}")

    # iterate through facility set, computer chip numbers
    for i in facilitySet:
        for z in range(1, computerChip + 1):
            Y[facilityDict[i], z] = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                            name = f"Y_{i}_{z}")

    # notify model the changes
    m.update()

    # set the objective function
    m.setObjective(quicksum(S[facilityDict[i], j, z] * X[facilityDict[i], j, z] for i 
                            in facilitySet for j in range(1, salesRegion + 1) for z in 
                            range(1, computerChip + 1)) + 
                quicksum(P[facilityDict[i], z] * Y[facilityDict[i], z] for i 
                            in facilitySet for z in range(1, computerChip + 1)))

    for i in facilitySet:
        facilityProductionConst[i] = m.addConstr(quicksum(
            Y[facilityDict[i], z] for z in range(1, computerChip + 1)
            ) <= 1000 * capacityDict[facilityDict[i]], name = f"production_{i}")
        
    for j in range(1, salesRegion + 1):
        for z in range(1, computerChip + 1):
            shipDemandConst[j, z] = m.addConstr(quicksum(
                X[facilityDict[i], j, z] for i in facilitySet) == 1000 * D[j, z],
                name = f"shipping_demand_{j}_{z}")

    for i in facilitySet:
        for z in range(1, computerChip + 1):
            productionShipConst[facilityDict[i], z] = m.addConstr(
                Y[facilityDict[i], z] >= quicksum(
                        X[facilityDict[i], j, z] for j in range(1, salesRegion + 1)), 
                        name = f"production_shipping_{i}_{z}")

    # for the fourth and fifth constraints, since Xijz and Yiz both have lower bound of
    # 0 on initialization, not adding them here
    # notify model the changes
    m.update()
    # trigger optimization
    m.optimize()
    # print the object function value
    print("Optimal Production and Distribution Costs = ", m.objVal)

# calculate the change in shipping/production cost after capacity increase in a facility
def cost_difference_calculator(oldTotalCost, newTotalCost):
    totalDiff = {}
    for k, v in oldTotalCost.items():
        if k not in totalDiff:
            totalDiff[k] = 0
        # if positive meaning it is more suboptimal, negative meaning it is more optimal
        totalDiff[k] += newTotalCost[k] - v
    return totalDiff

# calculate objective function value change before and after capacity increase in a facility
# if positive meaning it is more suboptimal, negative meaning it is more optimal
def objective_function_value_change(before, after):
    return after - before


# In[4]:


# optimal model total shipping cost per facility
optimalFacilityTotalShippingCost = total_shipping_cost_per_facility({}, shippingCosts, X)

# optimal model total production cost per facility
optimalFacilityTotalProductionCost = total_production_cost_per_facility({}, productionCosts, Y)

# increase Alexandria facility capacity by 50 (thousands)
ANewCapacity = capacity_increase_helper(50, productionCapacity, 0, 1)
CA = chip_production_per_facility(ANewCapacity, {})
print("Production Capacitiy After 50,000 Expansion: ", CA)

mA = Model()
XA = {}
YA = {}
AFacilityProductionConst = {}
AShipDemandConst = {}
AProductionShipConst = {}
ACapacityIncreaseModel = model_builder(mA, XA, YA, CA, AFacilityProductionConst, AShipDemandConst, AProductionShipConst)

# calculate new shipping cost difference
ANewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XA)
ATotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, ANewShippingCost)
print("Total Shipping Cost Change in Alexandria Facility with 50,000 Expansion: ", ATotalShippingCostChange)

# calculate new production cost difference
ANewProductionCost = total_production_cost_per_facility({}, productionCosts, YA)

ATotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, ANewProductionCost)
print("Total Production Cost change in Alexandria Facility with 50,000 Expansion: ", ATotalProductionCostChange)

# calculate objective value change
AObjValChange = objective_function_value_change(m.ObjVal, mA.ObjVal)
print("Change in Objective Function Value with 50,000 Expansion in Alexandria Facility: ", AObjValChange)


# increase Alexandria facility capacity by 100 (thousands)
A1NewCapacity = capacity_increase_helper(100, productionCapacity, 0, 1)
CA1 = chip_production_per_facility(A1NewCapacity, {})

mA1 = Model()
XA1 = {}
YA1 = {}
A1FacilityProductionConst = {}
A1ShipDemandConst = {}
A1ProductionShipConst = {}
A1CapacityIncreaseModel = model_builder(mA1, XA1, YA1, CA1, A1FacilityProductionConst, A1ShipDemandConst, A1ProductionShipConst)
print("Production Capacitiy After 100,000 Expansion: ", CA1)
# calculate new shipping cost difference
A1NewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XA1)
A1TotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, A1NewShippingCost)
print("Total Shipping Cost Change in Alexandria Facility with 100,000 Expansion: ", A1TotalShippingCostChange)

# calculate new production cost difference
A1NewProductionCost = total_production_cost_per_facility({}, productionCosts, YA1)
A1TotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, A1NewProductionCost)
print("Total Production Cost Change in Alexandria Facility with 100,000 Expansion: ", A1TotalProductionCostChange)

# calculate objective value change
A1ObjValChange = objective_function_value_change(m.ObjVal, mA1.ObjVal)
print("Change in Objective Function Value with 50,000 Expansion in Alexandria Facility: ", A1ObjValChange)


# In[5]:


# increase Richmond facility capacity by 50 (thousands)
RNewCapacity = capacity_increase_helper(50, productionCapacity, 1, 1)
CR = chip_production_per_facility(RNewCapacity, {})
print("Production Capacitiy after 50,000 Expansion: ", CR)

mR = Model()
XR = {}
YR = {}
RFacilityProductionConst = {}
RShipDemandConst = {}
RProductionShipConst = {}
RCapacityIncreaseModel = model_builder(mR, XR, YR, CR, RFacilityProductionConst, RShipDemandConst, RProductionShipConst)

# calculate new shipping cost difference
RNewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XR)
# calculate new production cost difference
RNewProductionCost = total_production_cost_per_facility({}, productionCosts, YR)

RTotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, RNewShippingCost)
print("Total Shipping Cost Change in Richmond Facility with 50,000 Expansion: ", RTotalShippingCostChange)

RTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RNewProductionCost)
print("Total Production Cost Change in Richmond Facility with 50,000 Expansion: ", RTotalProductionCostChange)

# calculate objective value change
RObjValChange = objective_function_value_change(m.ObjVal, mR.ObjVal)
print("Change in Objective Function Value in Richmond Facility with 50,000 Expansion: ", RObjValChange)


# increase Richmond facility capacity by 100 (thousands)
R1NewCapacity = capacity_increase_helper(100, productionCapacity, 1, 1)
CR1 = chip_production_per_facility(R1NewCapacity, {})

mR1 = Model()
XR1 = {}
YR1 = {}
R1FacilityProductionConst = {}
R1ShipDemandConst = {}
R1ProductionShipConst = {}
R1CapacityIncreaseModel = model_builder(mR1, XR1, YR1, CR1, R1FacilityProductionConst, R1ShipDemandConst, R1ProductionShipConst)
print("Production Capacitiy after 50,000 Expansion: ", CR1)
# calculate new shipping cost difference
R1NewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XR1)
# calculate new production cost difference
R1NewProductionCost = total_production_cost_per_facility({}, productionCosts, YR1)

R1TotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, R1NewShippingCost)
print("Total Shipping Cost Change in Richmond Facility with 100,000 Expansion: ", R1TotalShippingCostChange)

R1TotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, R1NewProductionCost)
print("Total Production Cost Change in Richmond Facility with 100,000 Expansion: ", R1TotalProductionCostChange)

# calculate objective value change
R1ObjValChange = objective_function_value_change(m.ObjVal, mR1.ObjVal)
print("Change in Objective Function Value in Richmond Facility with 100,000 Expansion: ", R1ObjValChange)


# In[6]:


# increase Norfolk facility capacity by 50 (thousands)
NNewCapacity = capacity_increase_helper(50, productionCapacity, 2, 1)
CN = chip_production_per_facility(NNewCapacity, {})
print("Production Capacitiy after 50,000 Expansion: ", CN)

mN = Model()
XN = {}
YN = {}
NFacilityProductionConst = {}
NShipDemandConst = {}
NProductionShipConst = {}
NCapacityIncreaseModel = model_builder(mN, XN, YN, CN, NFacilityProductionConst, NShipDemandConst, NProductionShipConst)

# calculate new shipping cost difference
NNewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XN)
# calculate new production cost difference
NNewProductionCost = total_production_cost_per_facility({}, productionCosts, YN)

NTotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, NNewShippingCost)
print("Total Shipping Cost Change in Norfolk Facility with 50,000 Expansion: ", NTotalShippingCostChange)

NTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, NNewProductionCost)
print("Total Production Cost Change in Norfolk Facility with 50,000 Expansion: ", NTotalProductionCostChange)

# calculate objective value change
NObjValChange = objective_function_value_change(m.ObjVal, mN.ObjVal)
print("Change in Objective Function Value in Norfolk Facility with 50,000 Expansion: ", NObjValChange)

# Increase Norfolk facility capacity by 100 (thousands)
N1NewCapacity = capacity_increase_helper(100, productionCapacity, 2, 1)
CN1 = chip_production_per_facility(N1NewCapacity, {})

mN1 = Model()
XN1 = {}
YN1 = {}
N1FacilityProductionConst = {}
N1ShipDemandConst = {}
N1ProductionShipConst = {}
NCapacityIncreaseModel = model_builder(mN1, XN1, YN1, CN1, N1FacilityProductionConst, N1ShipDemandConst, N1ProductionShipConst)
print("Production Capacitiy after 100,000 Expansion: ", CN1)
# calculate new shipping cost difference
N1NewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XN1)
# calculate new production cost difference
N1NewProductionCost = total_production_cost_per_facility({}, productionCosts, YN1)

N1TotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, N1NewShippingCost)
print("Total Shipping Cost Change in Norfolk Facility with 100,000 Expansion: ", N1TotalShippingCostChange)

N1TotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, N1NewProductionCost)
print("Total Production Cost Change in Norfolk Facility with 100,000 Expansion: ", N1TotalProductionCostChange)

# calculate objective value change
N1ObjValChange = objective_function_value_change(m.ObjVal, mN1.ObjVal)
print("Change in Objective Function Value in Norfolk Facility with 100,000 Expansion: ", N1ObjValChange)


# In[7]:


# increase Roanoke facility capacity by 50 (thousands)
RoNewCapacity = capacity_increase_helper(50, productionCapacity, 3, 1)
CRo = chip_production_per_facility(RoNewCapacity, {})
print("Production Capacitiy after 50,000 Expansion: ", CRo)

mRo = Model()
XRo = {}
YRo = {}
RoFacilityProductionConst = {}
RoShipDemandConst = {}
RoProductionShipConst = {}
RoCapacityIncreaseModel = model_builder(mRo, XRo, YRo, CRo, RoFacilityProductionConst, RoShipDemandConst, RoProductionShipConst)

# calculate new shipping cost difference
RoNewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XRo)
# calculate new production cost difference
RoNewProductionCost = total_production_cost_per_facility({}, productionCosts, YRo)

RoTotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, RoNewShippingCost)
print("Total Shipping Cost Change in Roanoke Facility with 50,000 Expansion: ", RoTotalShippingCostChange)

RoTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RoNewProductionCost)
print("Total Production Cost Change in Roanoke Facility with 50,000 Expansion: ", RoTotalProductionCostChange)

# calculate objective value change
RoObjValChange = objective_function_value_change(m.ObjVal, mRo.ObjVal)
print("Change in Objective Function Value in Roanoke Facility: ", RoObjValChange)

# increase Roanoke facility capacity by 100 (thousands)
Ro1NewCapacity = capacity_increase_helper(100, productionCapacity, 3, 1)
CRo1 = chip_production_per_facility(Ro1NewCapacity, {})

mRo1 = Model()
XRo1 = {}
YRo1 = {}
Ro1FacilityProductionConst = {}
Ro1ShipDemandConst = {}
Ro1ProductionShipConst = {}
Ro1CapacityIncreaseModel = model_builder(mRo1, XRo1, YRo1, CRo1, Ro1FacilityProductionConst, Ro1ShipDemandConst, Ro1ProductionShipConst)
print("Production Capacitiy after 100,000 Expansion: ", CRo1)
# calculate new shipping cost difference
Ro1NewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XRo1)
# calculate new production cost difference
Ro1NewProductionCost = total_production_cost_per_facility({}, productionCosts, YRo1)

Ro1TotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, Ro1NewShippingCost)
print("Total Shipping Cost Change in Roanoke Facility with 100,000 Expansion: ", RoTotalShippingCostChange)

Ro1TotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, Ro1NewProductionCost)
print("Total Production Cost Change in Roanoke Facility with 100,000 Expansion: ", RoTotalProductionCostChange)

# calculate objective value change
Ro1ObjValChange = objective_function_value_change(m.ObjVal, mRo1.ObjVal)
print("Change in Objective Function Value in Roanoke Facility: ", Ro1ObjValChange)


# In[8]:


# increase Charolottesville facility capacity by 50 (thousands)
CNewCapacity = capacity_increase_helper(50, productionCapacity, 4, 1)
CC = chip_production_per_facility(CNewCapacity, {})
print("Production Capacitiy after 50,000 Expansion:", CC)

mC = Model()
XC = {}
YC = {}
CFacilityProductionConst = {}
CShipDemandConst = {}
CProductionShipConst = {}
CCapacityIncreaseModel = model_builder(mC, XC, YC, CC, CFacilityProductionConst, CShipDemandConst, CProductionShipConst)

# calculate new shipping cost difference
CNewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XC)
# calculate new production cost difference
CNewProductionCost = total_production_cost_per_facility({}, productionCosts, YC)

CTotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, CNewShippingCost)
print("Total Shipping Cost Change in Charolottesville Facility with 50,000 Expansion: ", CTotalShippingCostChange)

CTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, CNewProductionCost)
print("Total Production Cost Change in Charolottesville Facility with 50,000 Expansion: ", CTotalProductionCostChange)

# calculate objective value change
CObjValChange = objective_function_value_change(m.ObjVal, mC.ObjVal)
print("Change in Objective Function Value in Charolottesville Facility: ", CObjValChange)


# increase Charolottesville facility capacity by 100 (thousands)
C1NewCapacity = capacity_increase_helper(100, productionCapacity, 4, 1)
C1C = chip_production_per_facility(C1NewCapacity, {})

mC1 = Model()
XC1 = {}
YC1 = {}
C1FacilityProductionConst = {}
C1ShipDemandConst = {}
C1ProductionShipConst = {}
C1CapacityIncreaseModel = model_builder(mC1, XC1, YC1, C1C, C1FacilityProductionConst, C1ShipDemandConst, C1ProductionShipConst)
print("Production Capacitiy after 100,000 Expansion: ", C1C)
# calculate new shipping cost difference
C1NewShippingCost = total_shipping_cost_per_facility({}, shippingCosts, XC1)
# calculate new production cost difference
C1NewProductionCost = total_production_cost_per_facility({}, productionCosts, YC1)

C1TotalShippingCostChange = cost_difference_calculator(optimalFacilityTotalShippingCost, C1NewShippingCost)
print("Total Shipping Cost Change in Charolottesville Facility with 100,000 Expansion: ", C1TotalShippingCostChange)

C1TotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, C1NewProductionCost)
print("Total Production Cost Change in Charolottesville Facility with 100,000 Expansion: ", C1TotalProductionCostChange)

# calculate objective value change
C1ObjValChange = objective_function_value_change(m.ObjVal, mC1.ObjVal)
print("Change in Objective Function Value in Charolottesville Facility: ", C1ObjValChange)


# In[9]:


# To answer question 3 - 

# build model for demand increase by 10% across all sales regions
m3 = Model()
# set objective function to be minimization
m3.modelSense = GRB.MINIMIZE
# iterate through facility set, sales region numbers and computer chip numbers
for i in facilitySet:
    for j in range(1, salesRegion + 1):
        for z in range(1, computerChip + 1):
            X[facilityDict[i], j, z] = m3.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                                name = f"X_{i}_{j}_{z}")

# iterate through facility set, computer chip numbers
for i in facilitySet:
    for z in range(1, computerChip + 1):
        Y[facilityDict[i], z] = m3.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                        name = f"Y_{i}_{z}")

# notify model the changes
m3.update()

# set the objective function
m3.setObjective(quicksum(S[facilityDict[i], j, z] * X[facilityDict[i], j, z] for i 
                        in facilitySet for j in range(1, salesRegion + 1) for z in 
                        range(1, computerChip + 1)) + 
                quicksum(P[facilityDict[i], z] * Y[facilityDict[i], z] for i 
                        in facilitySet for z in range(1, computerChip + 1)))

# first constraint ∑Yiz <= 1000Ci
facilityProductionConst = {}
for i in facilitySet:
     facilityProductionConst[i] = m3.addConstr(quicksum(
          Y[facilityDict[i], z] for z in range(1, computerChip + 1)
          ) <= 1000 * C[facilityDict[i]], name = f"production_{i}")

# second constraint ∑Xijz = 1000 * ((0.1 * Djz) + Djz)
shipDemandConst = {}
for j in range(1, salesRegion + 1):
     for z in range(1, computerChip + 1):
          shipDemandConst[j, z] = m3.addConstr(quicksum(
               # demand increase by 10% across all of the sales regions
               X[facilityDict[i], j, z] for i in facilitySet) == 1000 * (D[j, z] + (0.1 * D[j, z])),
               name = f"shipping_demand_{j}_{z}")

# third constraint Yiz >= ∑Xijz
productionShipConst = {}
for i in facilitySet:
     for z in range(1, computerChip + 1):
          productionShipConst[facilityDict[i], z] = m3.addConstr(
               Y[facilityDict[i], z] >= quicksum(
                    X[facilityDict[i], j, z] for j in range(1, salesRegion + 1)), 
                    name = f"production_shipping_{i}_{z}")

# for the fourth and fifth constraints, since Xijz and Yiz both have lower bound of
# 0 on initialization, not adding them here
# notify model the changes
m3.update()
# trigger optimization
m3.optimize()
# print the object function value
print("Optimal Production and Distribution Costs After 10% Demand Increase = ", m3.objVal)
# print the objective function value difference 
print("Optimal Production and Distribution Costs Diference = ", m3.objVal - m.objVal)


# In[10]:


# Question 5 - 

# preparation
# get the total production cost in each facility
print("Current Total Production Cost Per Facility: ", optimalFacilityTotalProductionCost)
maxFacility = max(optimalFacilityTotalProductionCost, key=optimalFacilityTotalProductionCost.get)
print("Facility With The Highest Total Production Cost: ", maxFacility, optimalFacilityTotalProductionCost[maxFacility])

# reduce production costs for all of the chips of the max facility by 15%
def new_production_cost_generator(facilityName, productionCosts):
    newProductionCosts = np.copy(productionCosts)
    for i in range(len(newProductionCosts)):
        if newProductionCosts[i][0] == facilityName:
            newProductionCosts[i][2] = newProductionCosts[i][2] - (newProductionCosts[i][2] * 0.15)
    return newProductionCosts

# generate model with production cost of 15% reduction
def model_builder(m, X, Y, capacityDict, facilityProductionConst, shipDemandConst, productionShipConst, P):
    # set objective function to be minimization
    m.modelSense = GRB.MINIMIZE

    # iterate through facility set, sales region numbers and computer chip numbers
    for i in facilitySet:
        for j in range(1, salesRegion + 1):
            for z in range(1, computerChip + 1):
                X[facilityDict[i], j, z] = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                                    name = f"X_{i}_{j}_{z}")

    # iterate through facility set, computer chip numbers
    for i in facilitySet:
        for z in range(1, computerChip + 1):
            Y[facilityDict[i], z] = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, 
                                            name = f"Y_{i}_{z}")

    # notify model the changes
    m.update()

    # set the objective function
    m.setObjective(quicksum(S[facilityDict[i], j, z] * X[facilityDict[i], j, z] for i 
                            in facilitySet for j in range(1, salesRegion + 1) for z in 
                            range(1, computerChip + 1)) + 
                quicksum(P[facilityDict[i], z] * Y[facilityDict[i], z] for i 
                            in facilitySet for z in range(1, computerChip + 1)))

    for i in facilitySet:
        facilityProductionConst[i] = m.addConstr(quicksum(
            Y[facilityDict[i], z] for z in range(1, computerChip + 1)
            ) <= 1000 * capacityDict[facilityDict[i]], name = f"production_{i}")
        
    for j in range(1, salesRegion + 1):
        for z in range(1, computerChip + 1):
            shipDemandConst[j, z] = m.addConstr(quicksum(
                X[facilityDict[i], j, z] for i in facilitySet) == 1000 * D[j, z],
                name = f"shipping_demand_{j}_{z}")

    for i in facilitySet:
        for z in range(1, computerChip + 1):
            productionShipConst[facilityDict[i], z] = m.addConstr(
                Y[facilityDict[i], z] >= quicksum(
                        X[facilityDict[i], j, z] for j in range(1, salesRegion + 1)), 
                        name = f"production_shipping_{i}_{z}")

    # for the fourth and fifth constraints, since Xijz and Yiz both have lower bound of
    # 0 on initialization, not adding them here
    # notify model the changes
    m.update()
    # trigger optimization
    m.optimize()
    # print the object function value
    print("Optimal Production and Distribution Costs = ", m.objVal)

def production_cost_mapper(productionCosts):
    # Piz = production cost per z chip in facility i ($)
    P = {}
    for p in productionCosts:
        # (i, z) = production cost in i, z
        P[facilityDict[p[0]], p[1]] = p[2]
    return P


# In[11]:


# if reduce production costs for all chips by 15% in Alexandria facility
mRA = Model()
XRA = {}
YRA = {}
RAFacilityProductionConst = {}
RAShipDemandConst = {}
RAProductionShipConst = {}
# generate new production cost array after reduction
RANewProductionCosts = new_production_cost_generator("Alexandria", productionCosts)
# map new production cost to model readable format
RAP = production_cost_mapper(RANewProductionCosts)
# generate Reduce Aleandria model
model_builder(mRA, XRA, YRA, C, RAFacilityProductionConst, 
                                        RAShipDemandConst, RAProductionShipConst, RAP)

# calculate new production cost difference
RANewProductionCost = total_production_cost_per_facility({}, RANewProductionCosts, YRA)
RATotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RANewProductionCost)
print("Total Production Costs Change After 15% Reduction in Alexandria: ", sum(RATotalProductionCostChange.values()))

# calculate objective value change
RAObjValChange = objective_function_value_change(m.ObjVal, mRA.ObjVal)
print("Change in Objective Function Value: ", RAObjValChange)


# In[12]:


# if reduce production costs for all chips by 15% in Richmond facility
mRR = Model()
XRR = {}
YRR = {}
RRFacilityProductionConst = {}
RRShipDemandConst = {}
RRProductionShipConst = {}
# generate new production cost array after reduction
RRNewProductionCosts = new_production_cost_generator("Richmond", productionCosts)
# map new production cost to model readable format
RRP = production_cost_mapper(RRNewProductionCosts)
# generate Reduce Aleandria model
model_builder(mRR, XRR, YRR, C, RRFacilityProductionConst, 
                                        RRShipDemandConst, RRProductionShipConst, RRP)

# calculate new production cost difference
RRNewProductionCost = total_production_cost_per_facility({}, RRNewProductionCosts, YRR)
RRTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RRNewProductionCost)
print("Total Production Costs Change After 15% Reduction in Richmond: ", sum(RRTotalProductionCostChange.values()))

# calculate objective value change
RRObjValChange = objective_function_value_change(m.ObjVal, mRR.ObjVal)
print("Change in Objective Function Value: ", RRObjValChange)


# In[13]:


# if reduce production costs for all chips by 15% in Norfolk facility
mRN = Model()
XRN = {}
YRN = {}
RNFacilityProductionConst = {}
RNShipDemandConst = {}
RNProductionShipConst = {}
# generate new production cost array after reduction
RNNewProductionCosts = new_production_cost_generator("Norfolk", productionCosts)
# map new production cost to model readable format
RNP = production_cost_mapper(RNNewProductionCosts)
# generate Reduce Aleandria model
model_builder(mRN, XRN, YRN, C, RNFacilityProductionConst, 
                                        RNShipDemandConst, RNProductionShipConst, RNP)

# calculate new production cost difference
RNNewProductionCost = total_production_cost_per_facility({}, RNNewProductionCosts, YRN)
RNTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RNNewProductionCost)
print("Total Production Cost Change After 15% Production Costs Reduction in Norfolk: ", sum(RNTotalProductionCostChange.values()))

# calculate objective value change
RNObjValChange = objective_function_value_change(m.ObjVal, mRN.ObjVal)
print("Change in Objective Function Value: ", RNObjValChange)


# In[14]:


# if reduce production costs for all chips by 15% in Roanoke facility
mRRO = Model()
XRRO = {}
YRRO = {}
RROFacilityProductionConst = {}
RROShipDemandConst = {}
RROProductionShipConst = {}
# generate new production cost array after reduction
RRONewProductionCosts = new_production_cost_generator("Roanoke", productionCosts)
# map new production cost to model readable format
RROP = production_cost_mapper(RRONewProductionCosts)
# generate Reduce Aleandria model
model_builder(mRRO, XRRO, YRRO, C, RROFacilityProductionConst, 
                                        RROShipDemandConst, RROProductionShipConst, RROP)

# calculate new production cost difference
RRONewProductionCost = total_production_cost_per_facility({}, RRONewProductionCosts, YRRO)
RROTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RRONewProductionCost)
print("Total Production Cost Change After 15% Production Costs Reduction in Roanoke: ", sum(RROTotalProductionCostChange.values()))

# calculate objective value change
RROObjValChange = objective_function_value_change(m.ObjVal, mRRO.ObjVal)
print("Change in Objective Function Value: ", RROObjValChange)


# In[15]:


# if reduce production costs for all chips by 15% in Charolottesville facility
mRC = Model()
XRC = {}
YRC = {}
RCFacilityProductionConst = {}
RCShipDemandConst = {}
RCProductionShipConst = {}
# generate new production cost array after reduction
RCNewProductionCosts = new_production_cost_generator("Charolottesville", productionCosts)
# map new production cost to model readable format
RCP = production_cost_mapper(RCNewProductionCosts)
# generate Reduce Aleandria model
model_builder(mRC, XRC, YRC, C, RCFacilityProductionConst, 
                                        RCShipDemandConst, RCProductionShipConst, RCP)

# calculate new production cost difference
RCNewProductionCost = total_production_cost_per_facility({}, RCNewProductionCosts, YRC)
RCTotalProductionCostChange = cost_difference_calculator(optimalFacilityTotalProductionCost, RCNewProductionCost)
print("Total Production Cost Change After 15% Production Costs Reduction in Charolottesville: ", sum(RCTotalProductionCostChange.values()))

# calculate objective value change
RCObjValChange = objective_function_value_change(m.ObjVal, mRC.ObjVal)
print("Change in Objective Function Value: ", RCObjValChange)


# 
