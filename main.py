import heapq

BLANK = 0

start = (
    (1, 6, 2),
    (5, 7, 8),
    (0, 4, 3)
)

goal = (
    (7, 8, 1),
    (6, 0, 2),
    (5, 4, 3)
)

# Precompute goal positions for each tile
goal_pos = {}
for r in range(3):
    for c in range(3):
        goal_pos[goal[r][c]] = (r, c)

# Windy movement cost for the tile that moves into the blank
# wind from east -> blows west:
# west = 1 (along), north/south = 2 (side), east = 3 (against)
MOVE_COST = {"west": 1, "east": 3, "north": 2, "south": 2}

def find_blank(state):
    for r in range(3):
        for c in range(3):
            if state[r][c] == BLANK:
                return r, c
    raise ValueError("No blank found")

def heuristic(state):
    windy_md = 0
    out_of_place = 0

    for r in range(3):
        for c in range(3):
            v = state[r][c]
            if v == BLANK:
                continue

            gr, gc = goal_pos[v]

            if (r, c) != (gr, gc):
                out_of_place += 1

            # horizontal: west steps cost 1, east steps cost 3
            if c > gc:
                windy_md += (c - gc) * 1        # move west
            else:
                windy_md += (gc - c) * 3        # move east

            # vertical: north/south both cost 2
            if r > gr:
                windy_md += (r - gr) * 2        # move north
            else:
                windy_md += (gr - r) * 2        # move south

    return windy_md + out_of_place

def neighbors_in_required_order(state):
    """
    Required child order is based on moving a non-blank tile into the neighboring blank:
    1) west neighboring tile, 2) north, 3) east, 4) south.
    That means we swap the blank with:
      (r, c-1), (r-1, c), (r, c+1), (r+1, c)
    and the MOVING TILE direction into the blank is:
      west tile -> moves east
      north tile -> moves south
      east tile -> moves west
      south tile -> moves north
    """
    br, bc = find_blank(state)

    # (delta to reach the tile being moved, direction the tile moves into blank)
    order = [
        (0, -1, "east"),   # tile west of blank moves east (against wind)
        (-1, 0, "south"),  # tile north of blank moves south (side wind)
        (0, 1, "west"),    # tile east of blank moves west (along wind)
        (1, 0, "north"),   # tile south of blank moves north (side wind)
    ]

    for dr, dc, tile_move_dir in order:
        tr, tc = br + dr, bc + dc
        if 0 <= tr < 3 and 0 <= tc < 3:
            grid = [list(row) for row in state]
            grid[br][bc], grid[tr][tc] = grid[tr][tc], grid[br][bc]
            new_state = tuple(tuple(row) for row in grid)
            step_cost = MOVE_COST[tile_move_dir]
            yield new_state, step_cost

def print_state(state, g, h, idx):
    for row in state:
        print("\t".join("-" if x == 0 else str(x) for x in row))
    print(f"{g}\t\t{h}")
    print(f"#{idx}")
    print()

def astar_expand_and_print(start, goal):
    # Frontier PQ items: (f, fifo_id, g, state)
    frontier = []
    fifo = 0

    g_best = {start: 0}      # hash table for best known g
    explored = set()         # hash set for expanded states

    h0 = heuristic(start)
    heapq.heappush(frontier, (h0, fifo, 0, start))

    expansion_idx = 0

    while frontier:
        f, _, g, state = heapq.heappop(frontier)

        # Lazy skip outdated entries
        if g_best.get(state) != g:
            continue
        if state in explored:
            continue

        explored.add(state)
        h = heuristic(state)

        # Print expansion
        print_state(state, g, h, expansion_idx)
        expansion_idx += 1

        if state == goal:
            break

        # Generate children in required order
        for child, step_cost in neighbors_in_required_order(state):
            ng = g + step_cost
            if child in explored and ng >= g_best.get(child, float("inf")):
                continue

            if ng < g_best.get(child, float("inf")):
                g_best[child] = ng
                fifo += 1
                nf = ng + heuristic(child)
                heapq.heappush(frontier, (nf, fifo, ng, child))

if __name__ == "__main__":
    astar_expand_and_print(start, goal)
