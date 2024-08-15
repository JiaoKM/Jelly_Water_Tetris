import taichi as ti
import numpy as np
import random
import time

ti.init(arch=ti.gpu)

paused = False
write_to_disk = True

quality = 1
n_grid = (64 * quality, 128 * quality)
dx, inv_dx = 1 / n_grid[1], n_grid[1]
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu)) # Lame parameters

max_num_particles = 1024 * 16
dim = 2
x = ti.Vector.field(dim, dtype=float)  # position
v = ti.Vector.field(dim, dtype=float)  # velocity
C = ti.Matrix.field(dim, dim, dtype=float)  # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=float)  # deformation gradient
material = ti.field(dtype=int)  # material id
Jp = ti.field(dtype=float)  # plastic deformation

ti.root.dynamic(ti.i, max_num_particles).place(x, v, C, F, material, Jp)
cur_num_particles = ti.field(ti.i32, shape=())

grid_v = ti.Vector.field(dim, dtype=float, shape=n_grid)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=n_grid)  # grid node mass

particle_radius = 2.5

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # p2g
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) +
                dt * C[p]) @ F[p]  # deformation gradient update
        h = ti.exp(
            10 *
            (1.0 -
             Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if material[p] >= 2:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 1:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[
                p] == 0:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 1:
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j][1] -= dt * 50  # gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid[0] - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j] = ti.Vector([0, 0])
            if j > n_grid[1] - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # g2p
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection

num_per_tetromino_square = 128
num_per_tetromino = num_per_tetromino_square * 4
staging_tetromino_x = ti.Vector.field(dim, dtype=float, shape=num_per_tetromino)
staging_tetromino_1 = ti.Vector.field(dim, dtype=float, shape=num_per_tetromino_square)
staging_tetromino_2 = ti.Vector.field(dim, dtype=float, shape=num_per_tetromino_square)
staging_tetromino_3 = ti.Vector.field(dim, dtype=float, shape=num_per_tetromino_square)
staging_tetromino_4 = ti.Vector.field(dim, dtype=float, shape=num_per_tetromino_square)
staging_list = [staging_tetromino_1, staging_tetromino_2, staging_tetromino_3, staging_tetromino_4]

class Block:
    def __init__(self, num_particles, width, color, material, init_particle, idx, x, init_pos, offset):
        self.num = num_particles
        self.w = width
        self.c = color
        self.mat = material
        self.start_particle = init_particle
        self.idx = idx
        self.x_field = x
        self.pos = init_pos
        self.offset = offset

    def fillBlock(self):
        shape = (num_per_tetromino_square, dim)
        x = np.zeros(shape=shape, dtype=np.float32)
        x += np.random.rand(*shape)
        x *= self.w
        self.x_field.from_numpy(x + np.array([self.pos, 0.8]) + self.offset * self.w)
    
@ti.kernel
def drop(mat: int):
    base = cur_num_particles[None]
    for i in staging_tetromino_x:
        bi = base + i
        x[bi] = staging_tetromino_x[i]
        material[bi] = mat
        v[bi] = ti.Matrix([0, -2])
        F[bi] = ti.Matrix([[1, 0], [0, 1]])
        Jp[bi] = 1
    cur_num_particles[None] += num_per_tetromino

@ti.kernel
def fillTetromino():
    for i in range(num_per_tetromino_square):
        staging_tetromino_x[i] = staging_tetromino_1[i]
        staging_tetromino_x[i + num_per_tetromino_square] = staging_tetromino_2[i]
        staging_tetromino_x[i + 2 * num_per_tetromino_square] = staging_tetromino_3[i]
        staging_tetromino_x[i + 3 * num_per_tetromino_square] = staging_tetromino_4[i]

@ti.kernel
def changeMaterial(start_particle: int):
    for i in range(num_per_tetromino_square):
        material[start_particle + i] = 0

@ti.kernel
def computeY(start_particle: int) -> ti.f32:
    Y = 0.0
    for i in range(num_per_tetromino_square):
        Y += x[start_particle + i][1]
    return Y / num_per_tetromino_square

def generateTetromino(offset, color, pos):
    blockList = []
    color_array[cur_num_particles[None]: cur_num_particles[None] + num_per_tetromino] += color
    for i in range(0, 4):
        block = Block(num_per_tetromino_square, 0.05, color, 2, cur_num_particles[None] + i * num_per_tetromino_square,
                      i, staging_list[i], pos, offset[i])
        block.fillBlock()
        blockList.append(block)
    return blockList


if __name__ == '__main__':
    gui = ti.GUI("Jelly water tetris", res=(384, 768), background_color=0x117081)
    colors = np.array([
            0xA6B5F7, 0xEEEEF0, 0xED553B, 0x3255A7, 0x6D35CB, 0xFE2E44,
            0x26A5A7, 0xEDE53B, 0x00FF72, 0xFFFFFF
        ], dtype=np.uint32)
    color_array = np.zeros((max_num_particles,), dtype=np.uint32)
    block_y_array = np.ones((int(max_num_particles / num_per_tetromino_square),), dtype=np.float32)
    changed_array = np.zeros((int(max_num_particles / num_per_tetromino_square),), dtype=np.uint32)
    
    tetromino_offsets = np.array([
        [[0, 0], [0, -1], [1, 0], [0, -2]],
        [[0, 0], [1, 1], [-1, 0], [1, 0]],
        [[0, 0], [0, -1], [-1, 0], [0, -2]],
        [[0, 0], [0, 1], [1, 0], [1, -1]],
        [[0, 0], [1, 0], [2, 0], [-1, 0]],
        [[0, 0], [0, 1], [1, 1], [1, 0]],
        [[0, 0], [-1, 0], [1, 0], [0, 1]],
    ])

    block_list = []
    start_flag = True

    t01 = t02 = time.time()
    for f in range(2000):
        if gui.get_event(ti.GUI.PRESS):
            ev_key = gui.event.key
            if ev_key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
            elif ev_key == 'p': paused = not paused
            elif ev_key == ti.GUI.SPACE:
                if cur_num_particles[None] + num_per_tetromino < max_num_particles:
                    offset_idx = random.randint(0, 6)
                    color_idx = random.randint(0, 9)
                    tetromino = generateTetromino(tetromino_offsets[offset_idx], colors[color_idx], random.random() * 0.38)
                    fillTetromino()
                    block_list.extend(tetromino)
                    drop(2)
        
        if not paused:
            t1 = time.time()
            if (t1 - t01) > 6 or start_flag:
                start_flag = False
                t01 = t1
                if cur_num_particles[None] + num_per_tetromino < max_num_particles:
                    offset_idx = random.randint(0, 6)
                    color_idx = random.randint(0, 9)
                    tetromino = generateTetromino(tetromino_offsets[offset_idx], colors[color_idx], random.random() * 0.38)
                    fillTetromino()
                    block_list.extend(tetromino)
                    drop(2)

            t2 = time.time()
            if (t2 - t02) > 2:
                t02 = t2
                changable_block = [[], [], [], [], [], []]
                for i in range(len(block_list)):
                    if changed_array[i] == 0:
                        y = computeY(block_list[i].start_particle)
                        block_y_array[i] = y
                        if y <= 0.05:
                            changable_block[0].append(i)
                        elif 0.05 < y <= 0.10:
                            changable_block[1].append(i)
                        elif 0.10 < y <= 0.15:
                            changable_block[2].append(i)
                        elif 0.15 < y <= 0.20:
                            changable_block[3].append(i)
                        elif 0.20 < y <= 0.25:
                            changable_block[4].append(i)
                        elif 0.25 < y <= 0.30:
                            changable_block[5].append(i)
                for list in changable_block:
                    if len(list) > 4:
                        for i in list:
                            changed_array[i] = 1
                            changeMaterial(block_list[i].start_particle)

            for s in range(int(2e-3 // dt)):
                    substep()
        
        gui.circles(x.to_numpy() * [[2, 1]], radius=particle_radius, color=color_array)
        
        if write_to_disk:
            gui.show(f'frames/{f:05d}.png')
        else:
            gui.show()
