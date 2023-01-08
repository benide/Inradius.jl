function experiment(N = 10, runs = 1:5,
                    diff_str = 10, diff_time = 2, diff_dt = 1e-2,
                    local_min_iters = 1000; x = rand(3, N))
    proj!(x)
    xs = [copy(x) for i in 1:Threads.nthreads()]
    best = [copy(x) for i in 1:Threads.nthreads()]
    dxs = [similar(x) for i in 1:Threads.nthreads()]
    Threads.@threads for run in runs
        thread = Threads.threadid()
        perturb_path!(dxs[thread], xs[thread], dLR!, proj!, diff_str, diff_time, diff_dt)
        local_minimum!(dxs[thread], xs[thread], LR, dLR!, proj!, iterations = local_min_iters)
        if L(xs[thread]) < L(best[thread])
            best[thread] .= xs[thread]
        end
    end
    best[argmin(L.(best))]
end

function experimentc(N = 10, runs = 1:5,
                     diff_str = 10, diff_time = 2, diff_dt = 1e-2,
                     local_min_iters = 1000; x = rand(3, N))
    proj!(x)
    xs = [copy(x) for i in 1:Threads.nthreads()]
    best = [copy(x) for i in 1:Threads.nthreads()]
    dxs = [similar(x) for i in 1:Threads.nthreads()]
    Threads.@threads for run in runs
        thread = Threads.threadid()
        perturb_path!(dxs[thread], xs[thread], dLcR!, proj!, diff_str, diff_time, diff_dt)
        local_minimum!(dxs[thread], xs[thread], LcR, dLcR!, proj!, iterations = local_min_iters)
        if Lc(xs[thread]) < L(best[thread])
            best[thread] .= xs[thread]
        end
    end
    best[argmin(Lc.(best))]
end
