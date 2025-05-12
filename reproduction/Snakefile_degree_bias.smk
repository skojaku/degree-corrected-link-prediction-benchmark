
params_degree_bias = {
    "nGroups": [2],
    "splitCriterion": ["degdeg"]
}
paramspace_degree_bias = to_paramspace(params_degree_bias)
DEG_BIAS_DIR = j(DERIVED_DIR, "degree-bias-analysis")

DEGREE_BIAS_NET_SCORE_FILE = j(
    DEG_BIAS_DIR,
    "{data}",
    f"score_basedOn~net_{paramspace_degree_bias.wildcard_pattern}_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_net_linkpred.wildcard_pattern}.csv",
)
DEGREE_BIAS_EMB_SCORE_FILE = j(
    DEG_BIAS_DIR,
    "{data}",
    f"score_basedOn~emb_{paramspace_degree_bias.wildcard_pattern}_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_emb.wildcard_pattern}.csv",
)

DEGREE_BIAS_BUDDY_SCORE_FILE = j(
    DEG_BIAS_DIR,
    "{data}",
    f"score_basedOn~buddy_{paramspace_degree_bias.wildcard_pattern}_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_buddy.wildcard_pattern}.csv",
)

DEGREE_BIAS_MLP_SCORE_FILE = j(
    DEG_BIAS_DIR,
    "{data}",
    f"score_basedOn~mlp_{paramspace_degree_bias.wildcard_pattern}_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_mlp.wildcard_pattern}.csv",
)

DEGREE_BIAS_LINEAR_SCORE_FILE = j(
    DEG_BIAS_DIR,
    "{data}",
    f"score_basedOn~linear_{paramspace_degree_bias.wildcard_pattern}_{paramspace_train_test_split.wildcard_pattern}_{paramspace_negative_edge_sampler.wildcard_pattern}_{paramspace_linear.wildcard_pattern}.csv",
)

DEG_BIAS_SCORE_FILE = j(
    RESULT_DIR,
    "result_degree_bias.csv",
)

rule eval_link_prediction_networks_degree_bias_toplogy:
    input:
        net_file=TRAIN_NET_FILE,
        edge_table_file=TARGET_EDGE_TABLE_FILE,
        pred_score_file=PRED_SCORE_NET_FILE,
    params:
        parameters=paramspace_degree_bias.instance,
    output:
        output_file=DEGREE_BIAS_NET_SCORE_FILE,
    script:
        "workflow/evaluate_degree_bias.py"

rule eval_link_prediction_networks_degree_bias_emb:
    input:
        net_file=TRAIN_NET_FILE,
        edge_table_file=TARGET_EDGE_TABLE_FILE,
        pred_score_file=PRED_SCORE_EMB_FILE,
    params:
        parameters=paramspace_degree_bias.instance,
    output:
        output_file=DEGREE_BIAS_EMB_SCORE_FILE,
    script:
        "workflow/evaluate_degree_bias.py"


rule eval_link_prediction_networks_degree_bias_buddy:
    input:
        net_file=TRAIN_NET_FILE,
        edge_table_file=TARGET_EDGE_TABLE_FILE,
        pred_score_file=PRED_SCORE_BUDDY_FILE,
    params:
        parameters=paramspace_degree_bias.instance,
    output:
        output_file=DEGREE_BIAS_BUDDY_SCORE_FILE,
    script:
        "workflow/evaluate_degree_bias.py"


rule eval_link_prediction_networks_degree_bias_mlp:
    input:
        net_file=TRAIN_NET_FILE,
        edge_table_file=TARGET_EDGE_TABLE_FILE,
        pred_score_file=PRED_SCORE_MLP_FILE,
    params:
        parameters=paramspace_degree_bias.instance,
    output:
        output_file=DEGREE_BIAS_MLP_SCORE_FILE,
    script:
        "workflow/evaluate_degree_bias.py"

rule eval_link_prediction_networks_degree_bias_linear:
    input:
        net_file=TRAIN_NET_FILE,
        edge_table_file=TARGET_EDGE_TABLE_FILE,
        pred_score_file=PRED_SCORE_LINEAR_FILE,
    params:
        parameters=paramspace_degree_bias.instance,
    output:
        output_file=DEGREE_BIAS_LINEAR_SCORE_FILE,
    script:
        "workflow/evaluate_degree_bias.py"


rule all_degree_bias:
    input:
        expand(DEGREE_BIAS_NET_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_net_linkpred, data=DATA_LIST),
        expand(DEGREE_BIAS_EMB_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_emb, data=DATA_LIST),
        #expand(DEGREE_BIAS_BUDDY_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_buddy, data=DATA_LIST),
        #expand(DEGREE_BIAS_MLP_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_mlp, data=DATA_LIST),
        expand(DEGREE_BIAS_LINEAR_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_linear, data=DATA_LIST),


rule concatenate_degree_bias_scores:
    input:
        expand(DEGREE_BIAS_NET_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_net_linkpred, data=DATA_LIST)+
        expand(DEGREE_BIAS_EMB_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_emb, data=DATA_LIST)+
        #expand(DEGREE_BIAS_BUDDY_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_buddy, data=DATA_LIST)+
        #expand(DEGREE_BIAS_MLP_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_mlp, data=DATA_LIST)+
        expand(DEGREE_BIAS_LINEAR_SCORE_FILE, **params_degree_bias, **params_train_test_split, **params_negative_edge_sampler, **params_linear, data=DATA_LIST),
    output:
        output_file=DEG_BIAS_SCORE_FILE,
    script:
        "workflow/concat-results.py"
