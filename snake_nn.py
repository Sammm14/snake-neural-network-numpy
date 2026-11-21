"""
Neural network that plays Snake -- NumPy only (no PyTorch / TensorFlow).

14 inputs -> 64 hidden (ReLU) -> 3 outputs (straight, right, left)
Q-learning (DQN-style): replay buffer, epsilon-greedy, target network.
Forward pass, loss, backprop and Adam are all written by hand.

Commands:
    python snake_nn.py train          # train the improved agent (14 inputs) -> weights.npz
    python snake_nn.py train basic    # train the baseline agent (11 inputs) -> weights_basic.npz
    python snake_nn.py eval           # fair evaluation: 100 greedy games per model
    python snake_nn.py gradcheck      # verify hand-written backprop vs numerical gradients
    python snake_nn.py play           # watch the AI
    python snake_nn.py versus         # you (arrow keys / WASD) vs the AI
    python snake_nn.py web            # write demo.html (runs in any browser)
    python snake_nn.py gif            # record snake.gif
    python snake_nn.py compare        # target network experiment (baseline features)
"""
import sys, random, json
from collections import deque
import numpy as np

GRID = 20
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up (clockwise)


# ----------------------------------------------------------------- environment
class Snake:
    def __init__(self, flood=True):
        self.flood = flood
        self.reset()

    def reset(self):
        self.d = 0
        self.body = [(GRID // 2, GRID // 2), (GRID // 2 - 1, GRID // 2), (GRID // 2 - 2, GRID // 2)]
        self.score = 0
        self.steps = 0
        self._food()
        return self.state()

    def _food(self):
        while True:
            self.food = (random.randrange(GRID), random.randrange(GRID))
            if self.food not in self.body:
                return

    def _hit(self, p):
        return not (0 <= p[0] < GRID and 0 <= p[1] < GRID) or p in self.body[:-1]

    def _ahead(self, d):
        dx, dy = DIRS[d % 4]
        return (self.body[0][0] + dx, self.body[0][1] + dy)

    def _space(self, p):
        """Flood fill from p: how much free room is there, as a fraction of snake length (capped at 1)."""
        if self._hit(p):
            return 0.0
        blocked, n = set(self.body[:-1]), len(self.body)
        seen, q, count = {p}, deque([p]), 0
        while q and count < n:
            x, y = q.popleft()
            count += 1
            for dx, dy in DIRS:
                w = (x + dx, y + dy)
                if 0 <= w[0] < GRID and 0 <= w[1] < GRID and w not in blocked and w not in seen:
                    seen.add(w)
                    q.append(w)
        return min(1.0, count / n)

    def state(self):
        hx, hy = self.body[0]
        fx, fy = self.food
        s = [
            self._hit(self._ahead(self.d)),      # danger ahead
            self._hit(self._ahead(self.d + 1)),  # danger right
            self._hit(self._ahead(self.d - 1)),  # danger left
            *[self.d == i for i in range(4)],    # heading
            fx < hx, fx > hx, fy < hy, fy > hy,  # food left / right / up / down
        ]
        if self.flood:                           # NEW: free space reachable after each move
            s += [self._space(self._ahead(self.d + k)) for k in (0, 1, -1)]
        return np.array(s, dtype=np.float32)

    def step(self, action):  # 0 straight, 1 turn right, 2 turn left
        self.d = (self.d + [0, 1, -1][action]) % 4
        new = self._ahead(self.d)
        self.steps += 1
        if self._hit(new) or self.steps > 100 * len(self.body):
            return self.state(), -10.0, True
        self.body.insert(0, new)
        if new == self.food:
            self.score += 1
            self.steps = 0
            self._food()
            return self.state(), 10.0, False
        self.body.pop()
        return self.state(), 0.0, False

    def step_abs(self, want):  # human control: absolute direction (180-degree turns ignored)
        return self.step({0: 0, 1: 1, 3: 2, 2: 0}[(want - self.d) % 4])


# --------------------------------------------------------------------- network
class Net:
    def __init__(self, n_in=14, n_hid=64, n_out=3, lr=1e-3, seed=0):
        r = np.random.default_rng(seed)
        self.p = {
            "W1": r.normal(0, np.sqrt(2 / n_in), (n_in, n_hid)), "b1": np.zeros(n_hid),
            "W2": r.normal(0, np.sqrt(2 / n_hid), (n_hid, n_out)), "b2": np.zeros(n_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t, self.lr = 0, lr

    def forward(self, x):
        h = np.maximum(0, x @ self.p["W1"] + self.p["b1"])  # h = max(0, W1 x + b1)
        return h @ self.p["W2"] + self.p["b2"], h

    def grads(self, x, a, target):
        """Loss and hand-written backprop gradients."""
        q, h = self.forward(x)
        B = len(x)
        idx = np.arange(B)
        err = q[idx, a] - target
        dq = np.zeros_like(q)
        dq[idx, a] = 2 * err / B                      # d loss / d q
        g = {"W2": h.T @ dq, "b2": dq.sum(0)}
        dh = (dq @ self.p["W2"].T) * (h > 0)          # backprop through ReLU
        g["W1"] = x.T @ dh
        g["b1"] = dh.sum(0)
        return np.mean(err ** 2), g

    def train_batch(self, x, a, target):
        loss, g = self.grads(x, a, target)
        self._adam(g)
        return loss

    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other):
        self.p = {k: v.copy() for k, v in other.p.items()}

    def save(self, f="weights.npz"):
        np.savez(f, **self.p)

    def load(self, f="weights.npz"):
        self.p = dict(np.load(f))

    @property
    def n_in(self):
        return self.p["W1"].shape[0]


# -------------------------------------------------------------------- training
def train(games=400, gamma=0.9, batch=128, eps_decay_games=120,
          use_target=True, sync_every=200, seed=0, flood=True, verbose=True):
    random.seed(seed); np.random.seed(seed)
    env = Snake(flood)
    n_in = 14 if flood else 11
    net, target_net = Net(n_in, seed=seed), Net(n_in, seed=seed)
    target_net.copy_from(net)
    mem, scores, steps = deque(maxlen=50_000), [], 0
    for g in range(games):
        eps = max(0.01, 1 - g / eps_decay_games)
        s, done = env.reset(), False
        while not done:
            if random.random() < eps:
                a = random.randrange(3)
            else:
                a = int(np.argmax(net.forward(s[None])[0]))   # a = argmax Q(s, a)
            s2, r, done = env.step(a)
            mem.append((s, a, r, s2, done))
            s, steps = s2, steps + 1
            if len(mem) >= batch:
                S, A, R, S2, D = map(np.array, zip(*random.sample(mem, batch)))
                nxt = target_net if use_target else net
                qn = nxt.forward(S2.astype(np.float32))[0].max(1)
                tgt = R + gamma * qn * (1 - D)                # Bellman target
                net.train_batch(S.astype(np.float32), A, tgt)
            if use_target and steps % sync_every == 0:
                target_net.copy_from(net)
        scores.append(env.score)
        if verbose and g % 25 == 0:
            print(f"game {g:4d}  score {env.score:2d}  best {max(scores):2d}  eps {eps:.2f}", flush=True)
    return net, scores


def moving_avg(x, w=20):
    return np.convolve(np.array(x, dtype=float), np.ones(w) / w, mode="valid")


def evaluate(net, games=100, seed=123, policy="greedy"):
    """Fair evaluation: no exploration, fixed seed, many games."""
    random.seed(seed)
    env, scores = Snake(net.n_in == 14), []
    for _ in range(games):
        s, done = env.reset(), False
        while not done:
            a = random.randrange(3) if policy == "random" else int(np.argmax(net.forward(s[None])[0]))
            s, _, done = env.step(a)
        scores.append(env.score)
    return scores


def cmd_train():
    basic = len(sys.argv) > 2 and sys.argv[2] == "basic"
    net, scores = train(flood=not basic)
    suffix = "_basic" if basic else ""
    net.save(f"weights{suffix}.npz")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(scores, alpha=.35, label="score per game")
    plt.plot(range(19, len(scores)), moving_avg(scores), label="20-game average")
    plt.xlabel("game"); plt.ylabel("apples eaten")
    plt.title("training: " + ("11 basic inputs" if basic else "14 inputs (with free-space)"))
    plt.legend(); plt.tight_layout(); plt.savefig(f"training{suffix}.png", dpi=150)
    print(f"best training game: {max(scores)} | saved weights{suffix}.npz and training{suffix}.png")


def cmd_eval():
    import os
    out = {}
    rnd = Net(14)  # untrained net is irrelevant; random policy baseline:
    out["random policy"] = evaluate(rnd, policy="random")
    for name, f in [("basic (11 inputs)", "weights_basic.npz"), ("improved (14 inputs)", "weights.npz")]:
        if os.path.exists(f):
            n = Net(); n.load(f)
            out[name] = evaluate(n)
    print(f"{'model':24s} {'mean':>6s} {'std':>6s} {'median':>7s} {'max':>5s}   (100 greedy games)")
    summary = {}
    for k, v in out.items():
        summary[k] = dict(mean=round(float(np.mean(v)), 1), std=round(float(np.std(v)), 1),
                          median=float(np.median(v)), max=int(max(v)))
        print(f"{k:24s} {summary[k]['mean']:6.1f} {summary[k]['std']:6.1f} {summary[k]['median']:7.1f} {summary[k]['max']:5d}")
    json.dump(summary, open("eval_results.json", "w"), indent=2)


def cmd_gradcheck():
    """Compare hand-written backprop with numerical (finite-difference) gradients."""
    rng = np.random.default_rng(1)
    net = Net(14, 16, 3)
    x = rng.normal(size=(8, 14)); a = rng.integers(0, 3, 8); t = rng.normal(size=8)
    _, g = net.grads(x, a, t)
    worst, h = 0.0, 1e-5
    for k, P in net.p.items():
        num = np.zeros_like(P)
        for i in np.ndindex(P.shape):
            old = P[i]
            P[i] = old + h; lp, _ = net.grads(x, a, t)
            P[i] = old - h; lm, _ = net.grads(x, a, t)
            P[i] = old
            num[i] = (lp - lm) / (2 * h)
        rel = np.max(np.abs(num - g[k]) / (np.abs(num) + np.abs(g[k]) + 1e-12))
        worst = max(worst, rel)
        print(f"{k}: max relative error {rel:.2e}")
    print("PASS" if worst < 1e-5 else "FAIL", f"(worst {worst:.2e})")


def cmd_compare(seeds=(0, 1), games=300):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    for name, ut in [("no target network", False), ("with target network", True)]:
        res[name] = [train(games=games, use_target=ut, seed=sd, flood=False, verbose=False)[1] for sd in seeds]
    plt.figure(figsize=(7, 4))
    for name, runs in res.items():
        plt.plot(range(19, games), np.mean([moving_avg(r) for r in runs], axis=0), label=name)
        print(name, "avg last 100:", round(float(np.mean([np.mean(r[-100:]) for r in runs])), 1))
    plt.xlabel("game"); plt.ylabel("apples eaten (20-game avg, mean of 2 seeds)")
    plt.title("Effect of a target network"); plt.legend(); plt.tight_layout(); plt.savefig("compare.png", dpi=150)


# ------------------------------------------------------------------ play modes
def load_net():
    net = Net()
    try:
        net.load()
    except FileNotFoundError:
        sys.exit("weights.npz not found -- run:  python3 snake_nn.py train")
    return net


def draw(scr, env, ox, C, pg):
    pg.draw.rect(scr, (15, 20, 25), (ox, 0, GRID * C, GRID * C))
    for x, y in env.body:
        pg.draw.rect(scr, (90, 220, 190), (ox + x * C, y * C, C - 2, C - 2))
    pg.draw.rect(scr, (240, 90, 100), (ox + env.food[0] * C, env.food[1] * C, C - 2, C - 2))


def cmd_play():
    import pygame as pg
    net = load_net()
    pg.init(); C = 24
    scr = pg.display.set_mode((GRID * C, GRID * C))
    clock, env = pg.time.Clock(), Snake(net.n_in == 14)
    s = env.reset()
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return
        s, _, done = env.step(int(np.argmax(net.forward(s[None])[0])))
        if done:
            s = env.reset()
        draw(scr, env, 0, C, pg)
        pg.display.set_caption(f"AI  score {env.score}")
        pg.display.flip(); clock.tick(15)


def cmd_versus():
    import pygame as pg
    net = load_net()
    pg.init(); C, W = 24, GRID * 24
    scr = pg.display.set_mode((2 * W + 10, W + 40))
    font, clock = pg.font.SysFont(None, 28), pg.time.Clock()
    me, ai = Snake(net.n_in == 14), Snake(net.n_in == 14)
    s = ai.reset(); me.reset()
    want, best = 0, {"me": 0, "ai": 0}
    keys = {pg.K_RIGHT: 0, pg.K_DOWN: 1, pg.K_LEFT: 2, pg.K_UP: 3, pg.K_d: 0, pg.K_s: 1, pg.K_a: 2, pg.K_w: 3}
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                return
            if e.type == pg.KEYDOWN and e.key in keys:
                want = keys[e.key]
        if me.step_abs(want)[2]:
            me.reset(); want = 0
        s, _, d2 = ai.step(int(np.argmax(net.forward(s[None])[0])))
        if d2:
            s = ai.reset()
        best["me"], best["ai"] = max(best["me"], me.score), max(best["ai"], ai.score)
        scr.fill((40, 45, 55))
        draw(scr, me, 0, C, pg); draw(scr, ai, W + 10, C, pg)
        scr.blit(font.render(f"YOU  score {me.score}  best {best['me']}", True, (230, 230, 230)), (8, W + 10))
        scr.blit(font.render(f"AI  score {ai.score}  best {best['ai']}", True, (230, 230, 230)), (W + 18, W + 10))
        pg.display.flip(); clock.tick(10)


def cmd_gif(frames=500):
    from PIL import Image, ImageDraw
    net = load_net()
    C, imgs = 16, []
    env = Snake(net.n_in == 14); s = env.reset()
    for _ in range(frames):
        s, _, done = env.step(int(np.argmax(net.forward(s[None])[0])))
        im = Image.new("RGB", (GRID * C, GRID * C), (15, 20, 25))
        dr = ImageDraw.Draw(im)
        for x, y in env.body:
            dr.rectangle([x * C, y * C, x * C + C - 2, y * C + C - 2], fill=(90, 220, 190))
        fx, fy = env.food
        dr.rectangle([fx * C, fy * C, fx * C + C - 2, fy * C + C - 2], fill=(240, 90, 100))
        imgs.append(im)
        if done:
            s = env.reset()
    imgs[0].save("snake.gif", save_all=True, append_images=imgs[1:], duration=60, loop=0)
    print("saved snake.gif")


HTML = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Neural network plays Snake</title>
<style>body{font-family:system-ui;background:#0f1419;color:#e6e6e6;text-align:center;margin:0;padding:20px}canvas{border:1px solid #333;max-width:100%}button{margin:4px;padding:8px 14px;border-radius:6px;border:0;background:#2a8;color:#fff;cursor:pointer}#i{font-size:20px;margin:8px}small{color:#889}</style>
<h2>Neural network plays Snake</h2><small>NumPy-trained Q-network (14 &rarr; 64 &rarr; 3) running live in your browser</small><br><br>
<canvas id=c width=480 height=480></canvas><div id=i></div>
<button id=m>Play yourself (arrow keys)</button><button id=r>Restart</button> Speed <input id=s type=range min=2 max=40 value=14>
<script>
const P=__W__,G=20,C=24,D=[[1,0],[0,1],[-1,0],[0,-1]],cx=c.getContext('2d');
let body,d,food,score,steps,best=0,human=false,want=0;
const K=p=>p[0]+','+p[1],inb=p=>p[0]>=0&&p[0]<G&&p[1]>=0&&p[1]<G;
function reset(){d=0;body=[[10,10],[9,10],[8,10]];score=0;steps=0;want=0;food_()}
function food_(){do{food=[Math.floor(Math.random()*G),Math.floor(Math.random()*G)]}while(body.some(b=>K(b)==K(food)))}
function ahead(k){const v=D[((k%4)+4)%4];return[body[0][0]+v[0],body[0][1]+v[1]]}
function hit(p){return!inb(p)||body.slice(0,-1).some(b=>K(b)==K(p))}
function space(p){if(hit(p))return 0;const bl=new Set(body.slice(0,-1).map(K)),n=body.length,seen=new Set([K(p)]),q=[p];let c=0;
while(q.length&&c<n){const u=q.shift();c++;for(const v of D){const w=[u[0]+v[0],u[1]+v[1]];if(!inb(w)||bl.has(K(w))||seen.has(K(w)))continue;seen.add(K(w));q.push(w)}}return Math.min(1,c/n)}
function state(){const[h0,h1]=body[0],[f0,f1]=food;
return[hit(ahead(d)),hit(ahead(d+1)),hit(ahead(d-1)),d==0,d==1,d==2,d==3,f0<h0,f0>h0,f1<h1,f1>h1].map(Number).concat([d,d+1,d-1].map(k=>space(ahead(k))))}
function fwd(x){const h=P.b1.map((b,j)=>{let s=b;for(let i=0;i<x.length;i++)s+=x[i]*P.W1[i][j];return Math.max(0,s)});
return P.b2.map((b,k)=>{let s=b;for(let j=0;j<h.length;j++)s+=h[j]*P.W2[j][k];return s})}
function step(a){d=(d+[0,1,3][a])%4;const n=ahead(d);steps++;
if(hit(n)||steps>100*body.length){best=Math.max(best,score);reset();return}
body.unshift(n);if(K(n)==K(food)){score++;steps=0;food_()}else body.pop()}
function draw(){cx.fillStyle='#0f1419';cx.fillRect(0,0,480,480);cx.fillStyle='#5adcbe';body.forEach(b=>cx.fillRect(b[0]*C,b[1]*C,C-2,C-2));
cx.fillStyle='#f05a64';cx.fillRect(food[0]*C,food[1]*C,C-2,C-2);i.textContent=(human?'YOU':'AI')+'  score '+score+'  best '+Math.max(best,score)}
function tick(){if(human)step([0,1,0,2][(want-d+4)%4]);else{const q=fwd(state());step(q.indexOf(Math.max(...q)))}draw();setTimeout(tick,1000/s.value)}
addEventListener('keydown',e=>{const k={ArrowRight:0,ArrowDown:1,ArrowLeft:2,ArrowUp:3,d:0,s:1,a:2,w:3}[e.key];if(k!==undefined){want=k;e.preventDefault()}});
m.onclick=()=>{human=!human;best=0;reset();m.textContent=human?'Watch the AI':'Play yourself (arrow keys)'};r.onclick=()=>{best=0;reset()};
reset();tick();
</script>"""


def cmd_web():
    net = load_net()
    if net.n_in != 14:
        sys.exit("demo needs the improved model: python3 snake_nn.py train")
    w = {k: np.round(v, 5).tolist() for k, v in net.p.items()}
    open("demo.html", "w").write(HTML.replace("__W__", json.dumps(w)))
    print("saved demo.html -- double-click it to open in your browser")


if __name__ == "__main__":
    cmds = {"train": cmd_train, "eval": cmd_eval, "gradcheck": cmd_gradcheck, "play": cmd_play,
            "versus": cmd_versus, "web": cmd_web, "gif": cmd_gif, "compare": cmd_compare}
    cmds.get(sys.argv[1] if len(sys.argv) > 1 else "train", cmd_train)()
