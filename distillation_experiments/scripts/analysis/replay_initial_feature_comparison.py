import json,pickle
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from brax import envs
from distillation.fit_feature_imitation import make_feature_cgp,feature_policy_action
from distillation.rollouts import rollout,valid_transition_mask,masked_return,sanitize_action
from distillation_experiments.scripts.experiments.operon_imitation import expression_to_jax
root=Path('distillation_experiments/artifacts/repertoires')
output=root/'initial_feature_comparison_evaluation'
output.mkdir(exist_ok=False)
environment=envs.get_environment('walker2d',backend='generalized')
corrections=[]
for model,mode,seed in [('CGP features + LR (k=8)','uniform',0),('CGP features + LR (k=8)','q_dagger',1),('Operon','uniform',3),('Operon','q_dagger',4)]:
 if model.startswith('CGP'):
  run=root/'cgp_feature_walker2d'/f'ann_initial_{mode}_k8_5seeds'
  row=json.loads((run/f'seed_{seed}/summary.json').read_text())
  config=json.loads((run/'config.json').read_text())
  with (run/f'seed_{seed}/final_individual.pickle').open('rb') as f: genotype=pickle.load(f)
  cgp=make_feature_cgp(config['n_inputs'],config['n_actions'],8,50)
  def action_fn(observation,key,step):
   action=feature_policy_action(genotype,cgp,observation);return action,action
  evaluation_seed=seed*100000+90000
 else:
  run=root/'operon_walker2d/operon_initial_both_5seeds'
  row=next(r for r in json.loads((run/'aggregate_summary.json').read_text())['variants'] if r['seed']==seed and r['mode']==mode)
  callables=[expression_to_jax(e,environment.observation_size) for e in row['expressions']]
  def action_fn(observation,key,step):
   action=sanitize_action(jnp.stack([jnp.asarray(fn(*observation)).reshape(()) for fn in callables]));return action,action
  evaluation_seed=seed*100000+(10000 if mode=='q_dagger' else 0)+90000
 def evaluate_one(keyseed):
  _,_,rewards,dones=rollout(environment,jax.random.key(keyseed),action_fn,1000)
  return rewards,dones
 rewards,dones=jax.vmap(evaluate_one)(evaluation_seed+jnp.arange(10))
 returns=jax.vmap(masked_return)(rewards,dones)
 masks=jax.vmap(valid_transition_mask)(dones)
 legacy=jnp.sum(rewards*masks,axis=1)
 invalid=jnp.sum((masks>0)&~jnp.isfinite(rewards),axis=1)
 row_out=dict(environment_key='walker2d',model=model,weighting='Uniform' if mode=='uniform' else 'Q-DAgger',seed=seed,
              source=str(run),original_reward=row['reward'],reward=float(jnp.mean(returns)),trajectory_returns=np.asarray(returns).tolist(),
              replay_legacy_returns=np.asarray(legacy).tolist(),invalid_reward_counts_before_termination=np.asarray(invalid).tolist(),
              first_terminal_steps=[int(x[0]) if len(x:=np.flatnonzero(d)) else None for d in np.asarray(dones)],
              first_nonfinite_reward_steps=[int(x[0]) if len(x:=np.flatnonzero(~np.isfinite(r))) else None for r in np.asarray(rewards)],
              evaluation_seed=evaluation_seed)
 corrections.append(row_out)
 label=('cgp' if model.startswith('CGP') else 'operon')+f'_{mode}_seed_{seed}'
 np.savez_compressed(output/f'{label}_trace.npz',rewards=np.asarray(rewards),dones=np.asarray(dones))
 (output/'corrections.json').write_text(json.dumps(corrections,indent=2)+'\n')
 print(label,'corrected reward',row_out['reward'],'invalid before termination',row_out['invalid_reward_counts_before_termination'],'legacy returns',row_out['replay_legacy_returns'],flush=True)
print('COMPLETE',flush=True)
