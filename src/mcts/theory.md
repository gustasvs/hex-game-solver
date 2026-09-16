

# NODE = position
# EDGE = action/move


# every node needs:
    N = visit count
    W = accumulated result/value

    Q = W / N

+1 = root player wins
-1 = root player loses

node.value means sum of outcomes from the perspective of the first player.

# Selection:

which child to investigate next?

for this we use UCT:

W / N = how good has this move been

C sqrt(ln(parent.N) / N) = how uncertain  am i about choosing this child?

`UCT = W / N + C sqrt(ln(parent.N) / N)`

constant C controls how aggresive is the exploration - small C explore less, larger C explore more, default is usually 1.4

`if a child is unvisited?` UCT = infinity - to ensure each move is looked at atleast once.

*** IMPORTANT: One MCTS iteration normally expands at most one new tree node ***

start at root
while position is not terminal:
    if node still has an untried move:
        create one child
        move into that child
        stop tree traversal
    otherwise:
        select best existing child using UCT
        move into it

tree phase:
persistent nodes/statistics
rollout phase:
temporary simulation
