import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

import torch

from scripts import phase3c_depth_protocol as protocol, phase3c_knockout25 as ko
from scripts import phase3c_scale as prefix,run_phase3c_scale as prefix_runner
from scripts.depth50 import run_phase3c_scale as runner
from scripts.phase3c_core import CONDITIONS,atomic_write,digest,edge_budget,read_json,support_record
from tests.test_phase3c_preflight import ToyText,toy_inputs


def cohort():
    keys = [f"pair{i:03d}" for i in range(50)]
    plan = {"fingerprint":"prefix","case_ids":keys,"representatives":keys[:2],
            "cohort_groups":{key:"original_pilot" if i < 8 else "newly_tested" for i,key in enumerate(keys)}}
    pairs = {key:{c:{"base_sample_id":i,"first_object_id":1+i%2,"phase3c_analysis_stratum":"rescue",
                      "prompt_variant":"swapped" if i == 10 else "original","correct_option":"A","eval_id":f"{key}_{c}"}
                 for c in CONDITIONS} for i,key in enumerate(keys)}
    return plan,pairs


class DepthGridTest(unittest.TestCase):
    def test_nine_locations_seven_new_and_no_duplicate_prefix(self):
        plan,_ = cohort()
        self.assertEqual(protocol.GRID,(0,4,8,12,16,20,24,28,32))
        self.assertEqual(protocol.LAYERS,(0,4,8,20,24,28,32))
        for stage,n in (("baseline",100),("identity",28),("patch",700)):
            tasks = protocol.tasks(plan,stage)
            self.assertEqual(len(tasks),n)
            self.assertEqual(len({r['task_id'] for r in tasks}),n)
        self.assertFalse({r['layer'] for r in protocol.tasks(plan,'patch')}&{12,16})
        self.assertEqual(len(protocol.tasks(plan,'patch'))+len(prefix.tasks(plan,'patch')),900)

    def test_adapter_does_not_change_existing_worker_globals(self):
        self.assertIs(prefix_runner.protocol,prefix)
        self.assertIs(runner.api.protocol,protocol)
        self.assertIsNot(prefix_runner,runner.api)
        self.assertEqual(prefix.LAYERS,(12,16))

    def test_full_observer_matches_existing_all_layer_capture(self):
        threads = torch.get_num_threads();torch.set_num_threads(1)
        try:
            model,inputs = ToyText(),toy_inputs()
            old = runner.ResidualSites(model,6,[1,2],[1,2])
            new = runner.DepthSites(model,6,[1,2],[1,2])
            with old.installed():a = model(*inputs)
            with new.installed():b = model(*inputs)
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(len(new.states),9)
            self.assertEqual(new.validate()['post_deepstack_layers'],[0,1,2])
            for key,state in new.states.items():
                self.assertTrue(torch.equal(state['vectors'],old.states[key]['vectors']))
            new.states.pop((0,'block_output'))
            with self.assertRaises(RuntimeError):new.validate()
        finally:torch.set_num_threads(threads)

    def test_complete_source_required_before_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix_root=Path(directory)/'prefix';status=Path(str(prefix_root)+'_lifecycle')/'job_status.json'
            args=SimpleNamespace(prefix_run=str(prefix_root))
            for value in ({'experiment_success':False,'backup_verified':True}, {'experiment_success':False,'backup_verified':False}):
                atomic_write(status,value)
                with patch.object(runner,'source_active',return_value=False),self.assertRaises(RuntimeError):
                    runner.wait_source(args,Mock())
            atomic_write(status,{'experiment_success':True,'backup_verified':True,'state':'needs_attention'})
            with patch.object(runner,'source_active',return_value=False):
                self.assertTrue(runner.wait_source(args,Mock())['experiment_success'])

    def test_storage_includes_extra_and_existing_compact_stores(self):
        value=protocol.storage_estimate()
        self.assertGreater(value['new_capture_upper_bytes'],prefix.capture_estimate({'capture_reuse_count':32},
            {'p':{'support_audit':{'supports':{'whole_event2':{'low_positions':list(range(512))}}}}})['new_capture_upper_bytes'])
        self.assertGreater(value['required_free_bytes'],2*value['new_capture_upper_bytes']+12*1024**3)

    def test_prefix_backup_recovery_never_waives_failed_scientific_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prefix_root=root/'prefix'
            args=SimpleNamespace(prefix_run=str(prefix_root),output_root=str(root/'depth'),backup_dir=str(root/'backups'))
            atomic_write(Path(str(prefix_root)+'_lifecycle')/'job_status.json',{'experiment_success':True,'backup_verified':False})
            with patch.object(runner,'source_active',return_value=False),patch.object(runner.api,'assert_idle'), \
                 patch.object(protocol.prefix,'load',return_value='context'), \
                 patch.object(prefix_runner,'verify_analysis',side_effect=ValueError('invalid')), \
                 patch.object(runner.api.shared,'create_backup') as backup,self.assertRaises(ValueError):
                runner.wait_source(args,Mock())
            backup.assert_not_called()
            with patch.object(runner,'source_active',return_value=False),patch.object(runner.api,'assert_idle'), \
                 patch.object(protocol.prefix,'load',return_value='context'),patch.object(prefix_runner,'verify_analysis'), \
                 patch.object(runner.api.shared,'create_backup',return_value=root/'bundle'), \
                 patch.object(runner.api.shared,'verify_backup',return_value={'complete':True,'reports_only':False,'available_activation_file_count':68}):
                result=runner.wait_source(args,Mock())
            self.assertTrue(result['backup_verified'])
            self.assertFalse(result['original_lifecycle_status']['backup_verified'])
            self.assertTrue((root/'depth/prefix_backup_handoff.json').exists())

    def test_extra_capture_saves_seven_sites_and_exactly_replays_prefix_two(self):
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);model,inputs=ToyText(),toy_inputs()
                row={'condition':CONDITIONS[0],'eval_id':'eval','phase3c_pair_id':'pair','correct_option':'A','archived_prediction':'B'}
                live={'row':row,'inputs':inputs,'input_tensor_sha256':{},'input_metadata':{},'archived_input_parity':{'matches':True}}
                mapping={'processor_records':{CONDITIONS[0]:{'input_ids':list(range(6)),'visual_positions':[1,2]}},
                         'support_audit':{'supports':{'whole_event2':support_record([1,2],[1,2],[1,2],[1,2],'whole')}}}
                engine=Mock();engine.text=model;engine.torch=torch;engine.model=model
                def forward(live):
                    hidden=model(*inputs)
                    return torch.stack((hidden[0,-1].sum()/100-2,torch.tensor(0.)))
                def decide(logits,row):
                    a,b=map(float,logits)
                    return {'margin':a-b,'correct_logit':a,'incorrect_logit':b,'correct_option':'A','prediction':'B','is_correct':False,'predicted_first_token_id':1}
                engine.forward.side_effect=forward;engine.decide.side_effect=decide
                old=runner.ResidualSites(model,6,[1,2],[1,2])
                with old.installed():logits=forward(live)
                path=root/'prefix/states.pt';index=path.with_name('index.json')
                runner.api.save_tensors(path,{'states':{k:v for k,v in old.states.items() if k[0] in (12,16) and k[1]=='block_output'},'logits':logits})
                atomic_write(index,{'artifact_type':'real','eval_id':'eval','input_tensor_sha256':{},'site_count':2,
                    'vectors_path':str(path),'vectors_sha256':runner.file_hash(path)})
                before={'eval_id':'eval','decision':decide(logits,row),'input_tensor_sha256':{},'record_sha256':'prefix-record',
                        'capture_index_path':str(index),'capture_index_sha256':runner.file_hash(index)}
                reference={'row':before,'index':read_json(index)}
                with patch('scripts.probe_attention_roi.standard_first_token',return_value=(1,logits)):
                    result=runner.capture_extra(engine,live,mapping,reference,root/'depth',{'fingerprint':'depth'})
                self.assertFalse(result['capture_reused']);self.assertTrue(all(result['source_replay'].values()))
                payload=runner.api.checked_payload(result,torch)
                self.assertEqual(set(payload['states']),{(layer,'block_output') for layer in protocol.LAYERS})
                self.assertNotIn((12,'block_output'),payload['states'])
                self.assertEqual(read_json(result['capture_index_path'])['site_count'],7)
        finally:torch.set_num_threads(threads)


class KO25Test(unittest.TestCase):
    def intent(self,whole=True):
        plan,pairs=cohort()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'analysis/case_knockout_decomposition.json'
            atomic_write(path,[])
            return ko.intent(plan,pairs,directory,whole),pairs

    def test_25_fixed_cases_original_eight_retained_mover_balanced(self):
        value,pairs=self.intent()
        self.assertEqual(len(value['case_ids']),25)
        self.assertTrue(set(value['retained_pilot_ids'])<=set(value['case_ids']))
        movers=[pairs[key][CONDITIONS[0]]['first_object_id'] for key in value['case_ids']]
        self.assertEqual(movers.count(1),13);self.assertEqual(movers.count(2),12)
        self.assertIn('pair010',value['case_ids'])

    def test_original_grid_and_whole_event_distinct_no_fake_background(self):
        value,_=self.intent()
        specs=ko.tasks(value,'fixed')
        self.assertEqual(len(specs),2925)
        self.assertEqual(len({r['task_id'] for r in specs}),2925)
        whole=[r for r in specs if r['key_group']=='whole_event2']
        self.assertEqual(len(whole),225)
        self.assertEqual({r['control'] for r in whole},{'whole_event_reference'})
        self.assertEqual({r['query_group'] for r in whole},{'options_all'})
        self.assertEqual(len(ko.tasks({**value,'whole_event2_enabled':False},'f')),2700)

    def test_causal_direction_and_all_head_budget(self):
        mapping={'processor_records':{CONDITIONS[0]:{'input_ids':list(range(8)),'group_positions':{'options_all':[6,7]}}},
            'support_audit':{'supports':{'whole_event2':{'low_positions':[1,2,3]}}}}
        q,k,b=ko.positions(mapping,CONDITIONS[0],{'key_group':'whole_event2'})
        self.assertEqual(q,[6,7]);self.assertEqual(k,[1,2,3])
        self.assertEqual(b['all_head_visible_causal_edges_per_layer'],2*3*32)

    def test_paired_decomposition_keeps_low_and_temporal_changes(self):
        plan,pairs=cohort();pair=pairs['pair000']
        spec={'pair_id':'pair000','query_group':'options_all','key_group':'whole_event2',
              'control':'whole_event_reference','window':[12,13,14,15]}
        def decision(m):return {'margin':m,'correct_logit':m+10,'incorrect_logit':10,'prediction':'A' if m>0 else 'B'}
        base={pair[c]['eval_id']:{'decision':decision(m)} for c,m in zip(CONDITIONS,(-1,2))}
        result={c:{'decision':decision(m),'edge_budget':edge_budget([6],[1,2],8)} for c,m in zip(CONDITIONS,(-.7,2.04))}
        row=ko.paired_row(spec,result,base,pairs)
        self.assertAlmostEqual(row['delta_M_low'],.3)
        self.assertAlmostEqual(row['delta_M_temporal'],.04)
        self.assertAlmostEqual(row['compression'],.26)
        self.assertAlmostEqual(row['advantage_KO'],2.74)

    def test_saved_active_mask_is_independently_validated(self):
        plan,pairs=cohort();pair_id='pair000';pairs={pair_id:pairs[pair_id]}
        plan={'fingerprint':'f','knockout':{'case_ids':[pair_id],'query_groups':['query_all'],
              'key_groups':['target_1'],'whole_event2_enabled':False}}
        budget=edge_budget([6],[1,2],8)
        route={'query_positions':[6],'target_key_positions':[1,2],'control_key_positions':[3,4]}
        mapping={'processor_records':{c:{'input_ids':list(range(8))} for c in CONDITIONS},
                 'support_audit':{'knockout_controls':{c:{'query_all':{'target_1':{'background':route}}} for c in CONDITIONS}}}
        mappings={pair_id:mapping}
        def decision(m):return {'margin':m,'correct_logit':m+10,'incorrect_logit':10,'prediction':'A' if m>0 else 'B'}
        baselines={pairs[pair_id][c]['eval_id']:{'record_sha256':c,'input_tensor_sha256':{},'decision':decision(m)}
                   for c,m in zip(CONDITIONS,(-1,2))}
        spec=ko.tasks(plan['knockout'],'f')[0]
        conditions={c:{'decision':decision(m),'query_positions':[6],'key_positions':[1,2],'edge_budget':budget,
            'input_tensor_sha256':{},'baseline_record_sha256':c,
            'mask_audit':{'enabled':True,'all_heads':32,'pre_softmax_mask':'negative_infinity',
                'layers':{str(i):{'budget':budget,'max_blocked_probability':0,'max_row_sum_error':0} for i in range(4)}}}
            for c,m in zip(CONDITIONS,(-1.2,1.5))}
        row={'task_id':spec['task_id'],'spec':spec,'plan_fingerprint':'f','passed':True,'reused':False,
             'conditions':conditions,'paired_outcome':ko.paired_row(spec,conditions,baselines,pairs)}
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output_root=directory);path=Path(directory)/'knockout/task_checkpoints'/f"{spec['task_id']}.json"
            atomic_write(path,runner.api.seal(row))
            with patch.object(ko,'old_rows',return_value={}):
                _,saved=runner.saved_ko(args,(plan,pairs,mappings,None,None),baselines=baselines)
                self.assertEqual(len(saved),1)
                row['conditions'][CONDITIONS[0]]['mask_audit']['layers']['0']['max_blocked_probability']=.1
                atomic_write(path,runner.api.seal(row))
                with self.assertRaises(ValueError):runner.saved_ko(args,(plan,pairs,mappings,None,None),baselines=baselines)


if __name__=='__main__':unittest.main()
