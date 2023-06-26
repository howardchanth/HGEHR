import dgl
import torch
from tqdm import tqdm

import pandas as pd

import pickle

from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.tasks import (
    drug_recommendation_mimic3_fn,
    mortality_prediction_mimic3_fn,
    length_of_stay_prediction_mimic3_fn,
    readmission_prediction_mimic3_fn,
    drug_recommendation_mimic4_fn,
    mortality_prediction_mimic4_fn,
    length_of_stay_prediction_mimic4_fn,
    readmission_prediction_mimic4_fn
)


class GraphConstructor:
    def __init__(self, config_graph):
        """

        :param config_graph:
        """
        self.config_graph = config_graph
        self.dataset_name = config_graph["dataset_name"]
        self.cache_path = config_graph["processed_path"]
        self.graph_path = config_graph["graph_output_path"]

        self.dataset = None
        self.graph = None
        self.mappings = None

        return

    def load_mimic(self):
        # Get mimic dataset from
        raw_path = self.config_graph["raw"]

        if "mimiciii" in raw_path:
            self.dataset = MIMIC3Dataset(
                root=raw_path,
                tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
                # tables=["DIAGNOSES_ICD"],
                code_mapping={},
                dev=False,
            )
        elif "mimiciv" in raw_path:
            self.dataset = MIMIC4Dataset(
                root=raw_path,
                tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
                code_mapping={},
                dev=False,
            )
        else:
            raise NotImplementedError

    def construct_graph(self):
        # Construct graph with the loaded datasets and tables
        graph_data = self.get_graph_data()
        self.graph = dgl.heterograph(graph_data)

    def initialize_features(self):

        """
        Initialize node features of the graph
        :return:
        """

        # Naive approach: randomly initialize the features

    def get_graph_data(self):

        # TODO: load gender and sex as attribute nodes to patients

        patients = self.dataset.patients

        # Dictionaries of indices
        patients_dict = {k: i for i, k in enumerate(patients.keys())}
        visits_set = set()
        diagnosis_set = set()
        procedures_set = set()
        prescriptions_set = set()
        labevents_set = set()

        patient_visit_edges = []
        visit_diagnosis_edges = []
        visit_procedure_edges = []
        visit_prescription_edges = []
        visit_labevent_edges = []

        for patient in tqdm(patients.values()):
            # Load visit numbers to map
            for visit in patient.visits.keys():
                visits_set.add(visit)
                patient_visit_edges.append((patient.patient_id, visit))

            # Traverse events
            for visit in patient.visits.values():
                ev_dict = visit.event_list_dict

                for table in visit.available_tables:
                    # Load diagnosis
                    if table.upper() == "DIAGNOSES_ICD":
                        for ev in ev_dict[table]:
                            diagnosis_set.add(ev.code)
                            visit_diagnosis_edges.append((ev.visit_id, ev.code))

                    if table.upper() == "PROCEDURES_ICD":
                        # Load procedure
                        for ev in ev_dict[table]:
                            procedures_set.add(ev.code)
                            visit_procedure_edges.append((ev.visit_id, ev.code))
                    if table.upper() == "PRESCRIPTIONS":
                        # Load prescriptions
                        for ev in ev_dict[table]:
                            prescriptions_set.add(ev.code)
                            visit_prescription_edges.append((ev.visit_id, ev.code))
                    if table.upper() == "LABEVENTS":
                        # Load prescriptions
                        for ev in ev_dict[table]:
                            labevents_set.add(ev.code)
                            visit_labevent_edges.append((ev.visit_id, ev.code))

        # Convert sets to dicts
        visits_dict = self.set_to_dict(visits_set)
        diagnosis_dict = self.set_to_dict(diagnosis_set)
        procedures_dict = self.set_to_dict(procedures_set)
        prescriptions_dict = self.set_to_dict(prescriptions_set)
        labevents_dict = self.set_to_dict(labevents_set)

        # Load graph indices
        patient_visit_edges = [(patients_dict[p], visits_dict[v]) for (p, v) in patient_visit_edges]
        visit_diagnosis_edges = [(visits_dict[v], diagnosis_dict[d]) for (v, d) in visit_diagnosis_edges]
        visit_labevent_edges = [(visits_dict[v], labevents_dict[l]) for (v, l) in visit_labevent_edges]
        visit_procedure_edges = [(visits_dict[v], procedures_dict[l]) for (v, l) in visit_procedure_edges]
        visit_prescription_edges = [(visits_dict[v], prescriptions_dict[l]) for (v, l) in visit_prescription_edges]

        # Create graph data
        graph_data = {}

        def update_edges(head, rel, tail, edges):

            graph_data.update(
                {
                    (head, rel, tail): (
                        torch.tensor([e[0] for e in edges]),
                        torch.tensor([e[1] for e in edges])
                    )
                }
            )

        update_edges("patient", "makes", "visit", patient_visit_edges)
        update_edges("visit", "diagnosed", "diagnosis", visit_diagnosis_edges)
        update_edges("visit", "prescribed", "prescription", visit_prescription_edges)
        update_edges("visit", "treated", "procedure", visit_procedure_edges)
        # update_edges("visit", "occurs", "labevent", visit_labevent_edges)

        # Save mappings
        self.mappings = {
            "patient": patients_dict,
            "visit": visits_dict,
            "diagnosis": diagnosis_dict,
            "procedure": procedures_dict,
            "prescription": prescriptions_dict,
            # "labevent": labevents_dict
        }

        with open(f'{self.graph_path}{self.dataset_name}_entity_mapping.pkl', 'wb') as outp:
            pickle.dump(self.mappings, outp, pickle.HIGHEST_PROTOCOL)

        return graph_data

    def set_tasks(self):

        mort_pred_samples, drug_rec_samples, los_samples, readm_samples = self.get_sample_datasets()

        vm = self.mappings["visit"]
        n_nodes = self.graph.num_nodes("visit")
        mort_labels = torch.zeros((n_nodes))

        # Assign mortality status
        mort_pred = {}
        for s in mort_pred_samples:
            visit_id = s["visit_id"]
            mort_pred.update({vm[visit_id]: s["label"]})

        # Assign drug recommendations
        drug_rec = {}
        all_drugs = {}
        for s in drug_rec_samples:
            visit_id = s["visit_id"]
            drug_rec.update({vm[visit_id]: s["drugs"]})

        # Assign length of stay
        los = {}
        for s in los_samples:
            visit_id = s["visit_id"]
            los.update({vm[visit_id]: s["label"]})

        # Assign readm_samples
        readm = {}
        for s in readm_samples:
            visit_id = s["visit_id"]
            readm.update({vm[visit_id]: s["label"]})

        labels = {
            "mort_pred": mort_pred,
            "drug_rec": drug_rec,
            "all_drugs": drug_rec_samples.get_all_tokens("drugs"),
            "los": los,
            "readm": readm
        }

        self.save_labels(labels)

    def get_sample_datasets(self):
        if "mimic3" in self.dataset_name:
            mort_pred_samples = self.dataset.set_task(mortality_prediction_mimic3_fn)
            drug_rec_samples = self.dataset.set_task(drug_recommendation_mimic3_fn)
            los_samples = self.dataset.set_task(length_of_stay_prediction_mimic3_fn)
            readm_samples = self.dataset.set_task(readmission_prediction_mimic3_fn)
        elif "mimic4" in self.dataset_name:
            mort_pred_samples = self.dataset.set_task(mortality_prediction_mimic4_fn)
            drug_rec_samples = self.dataset.set_task(drug_recommendation_mimic4_fn)
            los_samples = self.dataset.set_task(length_of_stay_prediction_mimic4_fn)
            readm_samples = self.dataset.set_task(readmission_prediction_mimic4_fn)
        else:
            raise ValueError

        return mort_pred_samples, drug_rec_samples, los_samples, readm_samples

    def get_mimic_dataset(self):
        with open(f'{self.cache_path}{self.dataset_name}', 'rb') as inp:
            unp = pickle.Unpickler(inp)
            mimic3_ds = unp.load()

        return mimic3_ds

    def save_mimic_dataset(self, mimic3_ds):
        # Save a copy to cache
        with open(f'{self.cache_path}{self.dataset_name}.pkl', 'wb') as outp:
            pickle.dump(mimic3_ds, outp, pickle.HIGHEST_PROTOCOL)

    def save_graph(self):
        with open(f'{self.graph_path}{self.dataset_name}.pkl', 'wb') as outp:
            pickle.dump(self.graph, outp, pickle.HIGHEST_PROTOCOL)

    def load_graph(self):
        with open(f'{self.graph_path}{self.dataset_name}', 'rb') as inp:
            unp = pickle.Unpickler(inp)
            g = unp.load()

        return g

    def save_labels(self, labels):
        with open(f'{self.graph_path}{self.dataset_name}_labels.pkl', 'wb') as outp:
            pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def set_to_dict(s):
        return {e: i for i, e in enumerate(s)}
