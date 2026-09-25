import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, RegularPolygon
from mcts.mcts import Node
from game_logic.hex import HexBoardState


def display_game(state: HexBoardState, root: Node) -> None:
    p1 = np.asarray(state.p1, dtype=bool)
    p2 = np.asarray(state.p2, dtype=bool)
    rows, columns = p1.shape

    row, column = np.indices((rows, columns))
    centers = np.column_stack((
        np.sqrt(3) * (column.ravel() + row.ravel() / 2),
        -1.5 * row.ravel(),
    ))
    angles = np.pi / 6 + np.arange(6) * np.pi / 3
    corners = np.column_stack((np.cos(angles), np.sin(angles)))
    hexagons = centers[:, None, :] + 0.98 * corners

    colors = np.full((rows * columns, 4), (1.0, 1.0, 1.0, 1.0))
    colors[p1.ravel()] = (0.85, 0.15, 0.15, 1.0)
    colors[p2.ravel()] = (0.15, 0.35, 0.85, 1.0)

    candidate_moves = []
    current_board = p1 if root.is_p1_turn() else p2
    for edge in root.children:
        child = edge.child
        child_board = np.asarray(
            child.state.p1 if root.is_p1_turn() else child.state.p2,
            dtype=bool,
        )
        changed_cells = np.flatnonzero(child_board & ~current_board)
        if changed_cells.size:
            candidate_moves.append((int(changed_cells[0]), edge.N, edge.W))

    best_move = None
    if candidate_moves:
        visits = np.fromiter((visits for _, visits, _ in candidate_moves), dtype=float)
        minimum = visits.min()
        spread = visits.max() - minimum
        strengths = (visits - minimum) / spread if spread else np.zeros_like(visits)
        teal = np.array((0.08, 0.68, 0.62, 1.0))
        for (cell_index, _, _), strength in zip(candidate_moves, strengths):
            colors[cell_index] = 1.0 + strength * (teal - 1.0)
        best_move = candidate_moves[int(np.argmax(visits))]

    figure, ax = plt.subplots()
    window = getattr(figure.canvas.manager, "window", None)
    set_geometry = getattr(window, "wm_geometry", None)
    move_window = getattr(window, "move", None)
    if callable(set_geometry):
        set_geometry("+100+100")
    elif callable(move_window):
        move_window(100, 100)
    ax.add_collection(PolyCollection(list(hexagons), facecolors=colors, edgecolors="black"))

    red_edges = []
    blue_edges = []
    gap = 0.14
    for cell in hexagons[:columns]:
        offset = np.array((0.0, gap))
        red_edges.extend(((cell[0] + offset, cell[1] + offset),
                          (cell[1] + offset, cell[2] + offset)))
    for cell in hexagons[-columns:]:
        offset = np.array((0.0, -gap))
        red_edges.extend(((cell[3] + offset, cell[4] + offset),
                          (cell[4] + offset, cell[5] + offset)))
    for cell in hexagons[::columns]:
        offset = gap * np.array((-np.sqrt(3) / 2, -0.5))
        blue_edges.extend(((cell[2] + offset, cell[3] + offset),
                           (cell[3] + offset, cell[4] + offset)))
    for cell in hexagons[columns - 1::columns]:
        offset = gap * np.array((np.sqrt(3) / 2, 0.5))
        blue_edges.extend(((cell[5] + offset, cell[0] + offset),
                           (cell[0] + offset, cell[1] + offset)))

    ax.add_collection(LineCollection(red_edges, colors="white", linewidths=9))
    ax.add_collection(LineCollection(blue_edges, colors="white", linewidths=9))
    ax.add_collection(LineCollection(red_edges, colors="#d92626", linewidths=5))
    ax.add_collection(LineCollection(blue_edges, colors="#2659d9", linewidths=5))
    for move in candidate_moves:
        cell_index, visit_count, value = move
        # marker = Circle(
        #     centers[cell_index],
        #     radius=0.45,
        #     facecolor="none",
        #     edgecolor="#263238",
        #     linewidth=2.5,
        #     zorder=5,
        # )
        # ax.add_patch(marker)
        ax.text(
            *centers[cell_index],
            str(round(visit_count, 2)),
            ha="center",
            va="center",
            color="#263238",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        )
    player_color = "#d92626" if root.is_p1_turn() else "#2659d9"
    ax.add_patch(RegularPolygon(
        (0.04, 1.1),
        6,
        radius=0.035,
        orientation=np.pi / 6,
        transform=ax.transAxes,
        facecolor=player_color,
        edgecolor="black",
        linewidth=1.5,
        clip_on=False,
        zorder=7,
    ))
    ax.text(
        0.085,
        1.1,
        "To move",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="#263238",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
        zorder=7,
    )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if window is None:
        plt.close(figure)
    else:
        plt.show()
