import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from PIL import Image
import time, os

def dct2(x): return dct(dct(x.T, norm='ortho').T, norm='ortho')
def idct2(x): return idct(idct(x.T, norm='ortho').T, norm='ortho')

def psnr(x, x0):
    N = x0.size
    peak = np.max(np.abs(x0))
    rmse = np.linalg.norm((x0 - x).ravel()) / np.sqrt(N)
    return np.inf if rmse == 0 else 20.0 * np.log10(peak / rmse)

def relerr(x, x0):
    return np.linalg.norm((x - x0).ravel()) / (np.linalg.norm(x0.ravel()) + 1e-30)

def make_mask(shape, frac, rng=None):
    rng = np.random.default_rng(rng)
    N = shape[0] * shape[1]
    k = int(np.round(frac * N))
    idx = rng.choice(N, size=k, replace=False)
    m = np.zeros(N, dtype=np.uint8); m[idx] = 1
    return m.reshape(shape)

def cg(matvec, b, x0=None, tol=1e-6, maxiter=None, M=None, callback=None):
    b = b.ravel()
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    z = r if M is None else M(r)
    p = z.copy()
    rz = np.dot(r, z)
    r0 = np.linalg.norm(r)
    if maxiter is None: maxiter = min(10 * b.size, 2000)
    if r0 <= max(1e-15, tol * (np.linalg.norm(b) + 1e-30)):
        return x, {'converged': True, 'iterations': 0, 'residual_norm': r0}
    for k in range(1, maxiter + 1):
        Ap = matvec(p)
        alpha = rz / (np.dot(p, Ap) + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        rn = np.linalg.norm(r)
        if callback: callback(k, rn)
        if rn <= tol * (r0 + 1e-30):
            return x, {'converged': True, 'iterations': k, 'residual_norm': rn}
        z = r if M is None else M(r)
        rz_new = np.dot(r, z)
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new
    return x, {'converged': False, 'iterations': maxiter, 'residual_norm': np.linalg.norm(r)}

def mm_reconstruct(obs, mask, lam, p=0.4, eps=1e-6, cg_tol=1e-6, mm_tol=1e-4, max_mm=200, show=False):
    x = obs.copy()
    Mmask = mask.astype(np.float64)
    obj_hist, rel_hist, cg_hist, t_hist = [], [], [], []
    b = obs.ravel()
    for it in range(max_mm):
        t0 = time.time()
        y = dct2(x)
        w = p * (eps + y**2) ** (p - 1)
        def A(v):
            v = v.reshape(obs.shape)
            return (Mmask * v + lam * idct2(w * dct2(v))).ravel()
        iters = []
        def cb(k, rn): iters.append(rn)
        sol, info = cg(A, b, x0=x.ravel(), tol=cg_tol, maxiter=min(10 * obs.size, 2000), callback=cb)
        x_new = sol.reshape(obs.shape)
        res = np.linalg.norm(Mmask * x_new - obs)**2
        y_new = dct2(x_new)
        pen = lam * np.sum((eps + y_new**2) ** p)
        obj = res + pen
        obj_hist.append(obj)
        rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12)
        rel_hist.append(rel)
        cg_hist.append(int(info.get('iterations', 0)))
        t_hist.append(time.time() - t0)
        if show: print(f"MM {it:3d} | J={obj:.3e} | rel={rel:.2e} | CG={cg_hist[-1]}")
        x = x_new
        if rel < mm_tol: break
    diag = {'objective_values': np.array(obj_hist),
            'relative_changes': np.array(rel_hist),
            'cg_iterations': np.array(cg_hist),
            'iteration_times': np.array(t_hist),
            'total_iterations': it + 1}
    return x, diag

def run_suite(paths,
              sample_fracs=[0.1, 0.2, 0.3, 0.5],
              p_vals=[0.3, 0.4, 0.5],
              lam_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1],
              snr_db=30.0,
              eps=1e-6,
              seed=0,
              outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for img_path in paths:
        im = Image.open(img_path).convert('L').resize((256, 256))
        x0 = np.asarray(im, dtype=np.float32) / 255.0
        base = os.path.splitext(os.path.basename(img_path))[0]

        for r in sample_fracs:
            mask = make_mask(x0.shape, r, rng)
            clean = mask * x0
            s_norm = np.linalg.norm(clean)
            n_target = s_norm / (10**(snr_db / 20.0) + 1e-12)
            n = rng.standard_normal(size=x0.shape) * mask
            n = (n_target / (np.linalg.norm(n) + 1e-30)) * n if np.linalg.norm(n) > 0 else 1e-12 * mask
            meas = clean + n

            per_p_best = {}
            sweep = []

            for p in p_vals:
                best = {'psnr': -np.inf}
                for lam in lam_grid:
                    xhat, diag = mm_reconstruct(meas, mask, lam, p=p, eps=eps,
                                                cg_tol=1e-6, mm_tol=1e-4, max_mm=200, show=False)
                    kval = psnr(xhat, x0)
                    rval = relerr(xhat, x0)
                    sweep.append((p, lam, kval, rval, diag))
                    if kval > best['psnr']:
                        best = {'lam': lam, 'psnr': kval, 'relerr': rval, 'xhat': xhat, 'diag': diag}
                per_p_best[p] = best

            # Save mask, noisy measurements
            prefix = f"{base}_r{int(r*100)}"
            plt.imsave(os.path.join(outdir, f"{prefix}_mask.png"), mask, cmap='gray')
            plt.imsave(os.path.join(outdir, f"{prefix}_noisy.png"), np.clip(meas, 0, 1), cmap='gray')

            # PSNR vs lambda for each p
            plt.figure(figsize=(6,4))
            for p in p_vals:
                L = np.array([lam for (pp, lam, kval, rval, d) in sweep if pp == p])
                P = np.array([kval for (pp, lam, kval, rval, d) in sweep if pp == p])
                idx = np.argsort(L)
                plt.plot(L[idx], P[idx], marker='o', label=f"p={p}")
            plt.xscale('log'); plt.xlabel('λ (log)'); plt.ylabel('PSNR (dB)')
            plt.title(f"{base}, sampling={r}")
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(outdir, f"{prefix}_psnr_vs_lambda.png")); plt.close()

            # Save per-(r,p) best reconstructions and residual maps
            for p in p_vals:
                b = per_p_best[p]
                xhat = b['xhat']
                lam = b['lam']
                plt.imsave(os.path.join(outdir, f"{prefix}_recon_p{p}_lam{lam}.png"),
                           np.clip(xhat, 0, 1), cmap='gray')
                resid = x0 - xhat
                plt.imsave(os.path.join(outdir, f"{prefix}_residual_p{p}_lam{lam}.png"),
                           resid, cmap='seismic')

            # Pick a representative case (best PSNR across p) for convergence + histogram plots
            p_star = max(p_vals, key=lambda p: per_p_best[p]['psnr'])
            bstar = per_p_best[p_star]
            xhat_star, lam_star, diag = bstar['xhat'], bstar['lam'], bstar['diag']

            # Convergence: objective
            plt.figure(figsize=(6,4))
            plt.plot(diag['objective_values'])
            plt.xlabel('MM iteration'); plt.ylabel('J(x)')
            plt.title(f"Convergence (p={p_star}, λ={lam_star})")
            plt.grid(True)
            plt.savefig(os.path.join(outdir, f"{prefix}_convergence_objective_p{p_star}_lam{lam_star}.png"))
            plt.close()

            # Convergence: relative change
            plt.figure(figsize=(6,4))
            plt.plot(diag['relative_changes'])
            plt.xlabel('MM iteration'); plt.ylabel('Relative change')
            plt.title(f"Rel. change (p={p_star}, λ={lam_star})")
            plt.grid(True)
            plt.savefig(os.path.join(outdir, f"{prefix}_convergence_relchange_p{p_star}_lam{lam_star}.png"))
            plt.close()

            # Convergence: CG iterations per MM step
            plt.figure(figsize=(6,4))
            plt.plot(diag['cg_iterations'], marker='o')
            plt.xlabel('MM iteration'); plt.ylabel('CG iterations')
            plt.title(f"CG per MM step (p={p_star}, λ={lam_star})")
            plt.grid(True)
            plt.savefig(os.path.join(outdir, f"{prefix}_convergence_cg_iters_p{p_star}_lam{lam_star}.png"))
            plt.close()

            # DCT histograms (original vs reconstruction of representative case)
            y0 = dct2(x0).ravel()
            yhat = dct2(xhat_star).ravel()
            plt.figure(figsize=(6,4))
            bins = 100
            plt.hist(np.abs(y0), bins=bins, alpha=0.6, label='|DCT(x*)|', density=True)
            plt.hist(np.abs(yhat), bins=bins, alpha=0.6, label='|DCT(x̂)|', density=True)
            plt.yscale('log'); plt.xlabel('magnitude'); plt.ylabel('density (log)')
            plt.title(f"DCT histograms (p={p_star}, λ={lam_star})")
            plt.legend()
            plt.savefig(os.path.join(outdir, f"{prefix}_dct_histograms_p{p_star}_lam{lam_star}.png"))
            plt.close()

            # CSV summary of best λ per p
            csv_path = os.path.join(outdir, f"{prefix}_summary.csv")
            with open(csv_path, 'w') as f:
                f.write("p,lambda,psnr,relerr\n")
                for p in p_vals:
                    b = per_p_best[p]
                    f.write(f"{p},{b['lam']},{b['psnr']:.6f},{b['relerr']:.6f}\n")

            print(f"{base} | r={r}: best per p -> " +
                  ", ".join([f"p={p}: λ={per_p_best[p]['lam']}, PSNR={per_p_best[p]['psnr']:.2f}dB"
                             for p in p_vals]))

if __name__ == "__main__":
    test_images = ["Images/cameraman.tif", "Images/lena.tif"]
    run_suite(test_images,
              sample_fracs=[0.1, 0.2, 0.3, 0.5],
              p_vals=[0.3, 0.4, 0.5],
              lam_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1],
              snr_db=30.0,
              eps=1e-6,
              seed=42,
              outdir='cs_reconstruction_results')
