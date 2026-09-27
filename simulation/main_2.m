%% main_5methods_threepanels.m
% Run three experiments:
%   1. varyK
%   2. varyNbad
%   3. varyBadFrac
%
% Compare five methods:
%   1. RF-ICA:
%      spectral kmeans on |A_c^T A_c| + Step 3 sign alignment + GM
%
%   2. RF-ICA w/o Step 3:
%      spectral kmeans on |A_c^T A_c| + raw GM, no sign alignment
%
%   3. F-ICA:
%      spectral kmeans on |A_c^T A_c| + Step 3 sign alignment + mean
%
%   4. simple mean:
%      directly aggregate local atoms with the same column index
%
%   5. simple median:
%      directly take coordinate-wise median of local atoms with the same column index

clear; clc; close all;
rng(1);
run_timer = tic;
output_dir = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% ===================== experiment configuration ==========================

repeats = 10;

base_params.r          = 10;
base_params.theta      = 0.10;
base_params.noise_std  = 0.00;

base_params.n_good     = 5000;
base_params.n_bad      = 100;
base_params.bad_frac   = 0.30;
base_params.K          = 30;

K_list        = [10, 30, 50, 70, 100];
n_bad_list    = [50, 70, 100, 300, 500, 1000];
bad_frac_list = 0:0.05:0.40;

exp_modes = {'varyK', 'varyNbad', 'varyBadFrac'};

method_names = { ...
    'SRF-ICA', ...
    'SRF-ICA-Noalignment', ...
    'F-ICA', ...
    'simple mean', ...
    'simple median'};

num_methods = numel(method_names);

all_results = struct();

%% ===================== run all three experiments ==========================

for e = 1:numel(exp_modes)
    exp_mode = exp_modes{e};
    params = base_params;

    switch exp_mode
        case 'varyK'
            sweep = K_list;
            result_name = 'K';

        case 'varyNbad'
            sweep = n_bad_list;
            result_name = 'n_bad';

        case 'varyBadFrac'
            sweep = bad_frac_list;
            result_name = 'bad_frac';

        otherwise
            error('Unknown exp_mode.');
    end

    errors = nan(num_methods, numel(sweep));
    errors_std = nan(num_methods, numel(sweep));
    errors_repeats = nan(repeats, num_methods, numel(sweep));

    for t = 1:numel(sweep)
        fprintf('\nRunning %s setting %d / %d ...\n', result_name, t, numel(sweep));

        switch exp_mode
            case 'varyK'
                params.K = sweep(t);

            case 'varyNbad'
                params.n_bad = sweep(t);

            case 'varyBadFrac'
                params.bad_frac = sweep(t);
        end

        err_rep = zeros(repeats, num_methods);

        for rep = 1:repeats
            fprintf('  repeat %d / %d (elapsed %.1f s)\n', rep, repeats, toc(run_timer));

            %% ----- generate global mixing matrix -----
            A_star = make_orth_dictionary(params.r);

            %% ----- local sample sizes -----
            n_local = params.n_good * ones(params.K, 1);
            num_bad = round(params.bad_frac * params.K);

            if num_bad > 0
                n_local(1:num_bad) = params.n_bad;
            end

            %% ----- generate local ICA estimators -----
            A_tilde = cell(params.K, 1);

            for k = 1:params.K
                Y_k = generate_client_data( ...
                    A_star, ...
                    n_local(k), ...
                    params.theta, ...
                    params.noise_std);

                A_tilde{k} = local_ica_varimax(Y_k, params.r);
            end

            %% ----- Method 1 and Method 3: RF-ICA and F-ICA -----
            % RF-ICA: spectral kmeans + Step 3 sign alignment + GM.
            % F-ICA: same clustering/sign alignment + mean.
            out_step3 = srf_ica_main_algorithm(A_tilde, params.r);

            %% ----- Method 2: no Step 3 control -----
            out_no_step3 = srf_ica_no_step3_control(A_tilde, params.r);

            %% ----- Method 4 and Method 5: simple index baselines -----
            [A_simple_mean, A_simple_median] = naive_index_baselines(A_tilde);

            %% ----- evaluate -----
            err_rep(rep, 1) = dict_error(out_step3.A_gm,        A_star);  % RF-ICA
            err_rep(rep, 2) = dict_error(out_no_step3.A_gm,     A_star);  % RF-ICA w/o Step 3
            err_rep(rep, 3) = dict_error(out_step3.A_mean,      A_star);  % F-ICA
            err_rep(rep, 4) = dict_error(A_simple_mean,         A_star);  % simple mean
            err_rep(rep, 5) = dict_error(A_simple_median,       A_star);  % simple median

            % Preserve individual repetitions for uncertainty bands.
            errors_repeats(rep, :, t) = err_rep(rep, :);
            rng_state = rng;
            save(fullfile(output_dir, 'checkpoint.mat'), ...
                'errors_repeats', 'exp_mode', 'sweep', 'params', ...
                'e', 't', 'rep', 'rng_state', '-v7');
        end

        assert(all(isfinite(err_rep), 'all'), 'Non-finite recovery error.');
        errors(:, t) = mean(err_rep, 1)';
        errors_std(:, t) = std(err_rep, 0, 1)';
        all_results(e).exp_mode = exp_mode;
        all_results(e).params = params;
        all_results(e).sweep = sweep;
        all_results(e).errors = errors;
        all_results(e).errors_std = errors_std;
        all_results(e).errors_repeats = errors_repeats;
        results.all_results = all_results;
        results.method_names = method_names;
        results.base_params = base_params;
        results.repeats = repeats;
        results.elapsed_seconds = toc(run_timer);
        save(fullfile(output_dir, 'simulation_results.mat'), 'results', '-v7');
    end

    all_results(e).exp_mode = exp_mode;
    all_results(e).params   = params;
    all_results(e).sweep    = sweep;
    all_results(e).errors   = errors;
end

%% ========================== plot saved results ==========================
% Plot separately with: python plot_results.py
% The plotting script produces three side-by-side panels with a shared
% legend and median / 25th-75th percentile bands over all 10 repetitions.
fprintf('Saved all repetitions to %s\n', fullfile(output_dir, 'simulation_results.mat'));
fprintf('Total elapsed time: %.1f s\n', toc(run_timer));

%% ========================================================================
%%                              functions
%% ========================================================================

function A = make_orth_dictionary(r)
    [Q, ~] = qr(randn(r, r), 0);
    A = normalize_columns(Q);
end

function Y = generate_client_data(A_star, n, theta, noise_std)
    r = size(A_star, 2);

    % Bernoulli-Gaussian sources.
    X = (rand(r, n) < theta) .* randn(r, n);

    Y = A_star * X + noise_std * randn(size(A_star, 1), n);
end

function A_hat = local_ica_varimax(Y, r)
    % Local ICA estimator using SVD + varimax / orthomax.

    [U, ~, V] = svd(Y, 'econ');

    r_eff = min([r, size(U, 2), size(V, 2)]);

    if r_eff == 1
        T = 1;
    else
        [~, T] = rotatefactors( ...
            V(:, 1:r_eff), ...
            'Method', 'orthomax', ...
            'Normalize', 'off', ...
            'Maxit', 1000);
    end

    A_hat = U(:, 1:r_eff) * T;
    A_hat = normalize_columns(A_hat);
end

function out = srf_ica_main_algorithm(A_tilde, r)
    % RF-ICA:
    %
    % Step 1: spectral embedding based on
    %         M = abs(A_c' * A_c) / sqrt(nbar)
    % Step 2: k-means on spectral embedding
    % Step 3: sign alignment within each estimated cluster
    % Step 4: GM and mean aggregation within aligned clusters

    %% ----- Step 1: collect raw local atoms -----
    A_c = [];
    atom_client = [];
    atom_local_idx = [];

    for k = 1:numel(A_tilde)
        Ak = normalize_columns(A_tilde{k});

        A_c = [A_c, Ak];

        atom_client = [atom_client, k * ones(1, size(Ak, 2))];
        atom_local_idx = [atom_local_idx, 1:size(Ak, 2)];
    end

    total_atoms = size(A_c, 2);
    nbar = total_atoms / r;

    %% ----- Step 1: construct M and spectral embedding -----
    M = abs(A_c' * A_c) / sqrt(nbar);
    M = (M + M') / 2;

    try
        [U_r, D] = eigs(M, r, 'largestreal');
    catch
        [U_r, D] = eigs(M, r, 'la');
    end

    [~, ord] = sort(diag(D), 'descend');
    U_r = U_r(:, ord);

    % Same as the paper: Q := U_r^T M.
    % Columns of Q correspond to atoms.
    Q = U_r' * M;

    %% ----- Step 2: k-means on columns of Q -----
    idx = kmeans(Q', r, ...
        'Display', 'off', ...
        'Replicates', 20);

    %% ----- Step 3 and Step 4: sign alignment + aggregation -----
    A_gm   = zeros(size(A_c, 1), r);
    A_mean = zeros(size(A_c, 1), r);

    clusters = cell(r, 1);
    A_c_aligned = A_c;

    for a = 1:r
        members = find(idx == a);
        clusters{a} = members;

        if isempty(members)
            warning('Empty cluster encountered in RF-ICA.');
            continue;
        end

        X = A_c(:, members);

        %% ----- Step 3: align signs within this estimated cluster -----
        [u_hat, ~, ~] = svds(X, 1);

        signs = sign(u_hat' * X);
        signs(signs == 0) = 1;

        X_aligned = X .* signs;
        A_c_aligned(:, members) = X_aligned;

        %% ----- Step 4: aggregate within aligned cluster -----
        A_mean(:, a) = mean(X_aligned, 2);
        A_gm(:, a)   = geometric_median_weiszfeld(X_aligned, 1e-8, 500);
    end

    A_mean = normalize_columns(A_mean);
    A_gm   = normalize_columns(A_gm);

    out.A_c            = A_c;
    out.A_c_aligned    = A_c_aligned;
    out.M              = M;
    out.Q              = Q;
    out.idx            = idx;
    out.clusters       = clusters;
    out.A_mean         = A_mean;
    out.A_gm           = A_gm;
    out.atom_client    = atom_client;
    out.atom_local_idx = atom_local_idx;
end

function out = srf_ica_no_step3_control(A_tilde, r)
    % Control method:
    %
    % Step 1: spectral embedding based on
    %         M = abs(A_c' * A_c) / sqrt(nbar)
    % Step 2: k-means on spectral embedding
    % Step 3: skipped
    % Step 4: raw GM and raw mean aggregation
    %
    % This control tests the necessity of within-cluster sign alignment.

    %% ----- collect raw local atoms -----
    A_c = [];
    atom_client = [];
    atom_local_idx = [];

    for k = 1:numel(A_tilde)
        Ak = normalize_columns(A_tilde{k});

        A_c = [A_c, Ak];

        atom_client = [atom_client, k * ones(1, size(Ak, 2))];
        atom_local_idx = [atom_local_idx, 1:size(Ak, 2)];
    end

    total_atoms = size(A_c, 2);
    nbar = total_atoms / r;

    %% ----- construct M and spectral embedding -----
    M = abs(A_c' * A_c) / sqrt(nbar);
    M = (M + M') / 2;

    try
        [U_r, D] = eigs(M, r, 'largestreal');
    catch
        [U_r, D] = eigs(M, r, 'la');
    end

    [~, ord] = sort(diag(D), 'descend');
    U_r = U_r(:, ord);

    % Same as paper: Q := U_r^T M.
    Q = U_r' * M;

    %% ----- k-means on columns of Q -----
    idx = kmeans(Q', r, ...
        'Display', 'off', ...
        'Replicates', 20);

    %% ----- raw aggregation without Step 3 -----
    A_gm   = zeros(size(A_c, 1), r);
    A_mean = zeros(size(A_c, 1), r);

    clusters = cell(r, 1);

    for a = 1:r
        members = find(idx == a);
        clusters{a} = members;

        if isempty(members)
            warning('Empty cluster encountered in no-Step-3 control.');
            continue;
        end

        X = A_c(:, members);

        % No sign alignment here.
        A_mean(:, a) = mean(X, 2);
        A_gm(:, a)   = geometric_median_weiszfeld(X, 1e-8, 500);
    end

    A_mean = normalize_columns(A_mean);
    A_gm   = normalize_columns(A_gm);

    out.A_c            = A_c;
    out.M              = M;
    out.Q              = Q;
    out.idx            = idx;
    out.clusters       = clusters;
    out.A_mean         = A_mean;
    out.A_gm           = A_gm;
    out.atom_client    = atom_client;
    out.atom_local_idx = atom_local_idx;
end

function [A_mean, A_median] = naive_index_baselines(A_tilde)
    % Simple baselines:
    % directly aggregate atoms with the same local column index.
    %
    % No permutation matching.
    % No sign alignment.
    %
    % A_mean(:,j)   = mean_k A_tilde{k}(:,j)
    % A_median(:,j) = coordinate-wise median_k A_tilde{k}(:,j)

    K = numel(A_tilde);
    [d, r] = size(A_tilde{1});

    A_stack = zeros(d, r, K);

    for k = 1:K
        A_stack(:, :, k) = normalize_columns(A_tilde{k});
    end

    A_mean = mean(A_stack, 3);

    A_median = zeros(d, r);
    for j = 1:r
        A_median(:, j) = median(squeeze(A_stack(:, j, :)), 2);
    end

    A_mean   = normalize_columns(A_mean);
    A_median = normalize_columns(A_median);
end

function g = geometric_median_weiszfeld(X, tol, maxit)
    % Ordinary geometric median:
    %
    %       min_g sum_j ||g - x_j||_2.
    %
    % X is d x N, with columns as points.

    if nargin < 2
        tol = 1e-8;
    end

    if nargin < 3
        maxit = 500;
    end

    X = normalize_columns(X);

    g = mean(X, 2);

    for it = 1:maxit
        diffs = X - g;
        dists = sqrt(sum(diffs.^2, 1));

        if any(dists < tol)
            g = X(:, find(dists == min(dists), 1));
            return;
        end

        w = 1 ./ max(dists, tol);
        g_new = (X * w') / sum(w);

        if norm(g_new - g) < tol * max(1, norm(g))
            g = g_new;
            return;
        end

        g = g_new;
    end
end

function A = normalize_columns(A)
    nrm = sqrt(sum(A.^2, 1));
    nrm(nrm < 1e-12) = 1;
    A = A ./ nrm;
end

function err = dict_error(A_hat, A_star)
    % Signed-permutation-invariant dictionary recovery error.
    %
    % This matching is only for evaluation.

    A_hat  = normalize_columns(A_hat);
    A_star = normalize_columns(A_star);

    G = abs(A_hat' * A_star);
    pairs = best_matching_from_similarity(G);

    A1 = A_hat(:, pairs(:, 1));
    A2 = A_star(:, pairs(:, 2));

    signs = sign(sum(A1 .* A2, 1));
    signs(signs == 0) = 1;

    A1 = A1 .* signs;

    err = norm(A1 - A2, 'fro') / sqrt(size(A_star, 2));
end

function pairs = best_matching_from_similarity(G)
    % Maximize total similarity.
    % Output:
    %   pairs(:,1): row indices
    %   pairs(:,2): column indices

    r1 = size(G, 1);
    r2 = size(G, 2);

    if exist('matchpairs', 'file') == 2
        C = max(G(:)) - G;
        pairs = matchpairs(C, 1e9);
        return;
    end

    % Greedy fallback if matchpairs is unavailable.
    m = min(r1, r2);
    pairs = zeros(m, 2);

    rows_left = 1:r1;
    cols_left = 1:r2;

    for t = 1:m
        subG = G(rows_left, cols_left);

        [~, idx] = max(subG(:));
        [i_sub, j_sub] = ind2sub(size(subG), idx);

        pairs(t, 1) = rows_left(i_sub);
        pairs(t, 2) = cols_left(j_sub);

        rows_left(i_sub) = [];
        cols_left(j_sub) = [];
    end
end
