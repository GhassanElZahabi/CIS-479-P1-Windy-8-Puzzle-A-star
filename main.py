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

# Wind costs (tile moves into blank): west=1, north/south=2, east=3

MOVE_COST = {"west": 1, "east": 3, "north": 2, "south": 2}


def find_blank(state):
    for r in range(3):
        for c in range(3):
            if state[r][c] == BLANK:
                return r, c
    raise ValueError("No blank found")


def windy_manhattan(state):
    """ Sum_{i=1..8} h_i(n): windy Manhattan only """
    h = 0
    for r in range(3):
        for c in range(3):
            v = state[r][c]
            if v == BLANK:
                continue
            gr, gc = goal_pos[v]

            # horizontal: west steps cost 1, east steps cost 3
            if c > gc:
                h += (c - gc) * 1        # move west
            else:
                h += (gc - c) * 3        # move east

            # vertical: north/south both cost 2
            if r > gr:
                h += (r - gr) * 2        # move north
            else:
                h += (gr - r) * 2        # move south
    return h


def out_of_place(state):
    """ h^(n): number of misplaced tiles (excluding blank) """
    misplaced = 0
    for r in range(3):
        for c in range(3):
            v = state[r][c]
            if v == BLANK:
                continue
            if goal[r][c] != v:
                misplaced += 1
    return misplaced


def heuristic(state):

    """
    Assignment heuristic: windy Manhattan + misplaced tiles (excluding blank)

    """

    return windy_manhattan(state) + out_of_place(state)


def neighbors_in_required_order(state):
    """
    Child order based on moving a tile into the blank:
    1) west tile, 2) north tile, 3) east tile, 4) south tile

    This means blank swaps with:
      (r, c-1), (r-1, c), (r, c+1), (r+1, c)

    And the moved tile direction into the blank is:
      west tile -> moves east
      north tile -> moves south
      east tile -> moves west
      south tile -> moves north
    """
    br, bc = find_blank(state)

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


def print_state_like_assignment(state, g, h, idx):
    for row in state:
        print("\t".join("-" if x == 0 else str(x) for x in row))
    print(f"{g}\t\t{h}")
    print(f"#{idx}")
    print()


def reconstruct_path(parent, g_cost, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    total_cost = g_cost[goal]
    return path, total_cost


def astar_optimal_path_and_print(start, goal):
    frontier = []
    fifo = 0

    parent = {start: None}
    g_best = {start: 0}
    explored = set()

    heapq.heappush(frontier, (heuristic(start), fifo, 0, start))

    while frontier:
        f, _, g, state = heapq.heappop(frontier)

        # Skip outdated entries
        if g_best.get(state) != g:
            continue
        if state in explored:
            continue
        explored.add(state)

        if state == goal:
            path, total_cost = reconstruct_path(parent, g_best, goal)

            print("solution path found by A* using the assignment heuristic:\n")
            for idx, s in enumerate(path):
                print_state_like_assignment(s, g_best[s], heuristic(s), idx)

            print(f"TOTAL COST = {total_cost}")
            return path, total_cost

        for child, step_cost in neighbors_in_required_order(state):
            ng = g + step_cost

            # best-g check
            if ng < g_best.get(child, float("inf")):
                g_best[child] = ng
                parent[child] = state
                fifo += 1
                nf = ng + heuristic(child)
                # Frontier items: (f, fifo, g, state)
                heapq.heappush(frontier, (nf, fifo, ng, child))

    print("No solution found.")
    return None, None


if __name__ == "__main__":
    astar_optimal_path_and_print(start, goal)
