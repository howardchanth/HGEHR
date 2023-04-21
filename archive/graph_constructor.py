import dgl
import torch
from tqdm import tqdm

import pandas as pd

import pickle

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import (
    drug_recommendation_mimic3_fn,
    mortality_prediction_mimic3_fn,
    length_of_stay_prediction_mimic3_fn,
    readmission_prediction_mimic3_fn
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

        self.mimic3_ds = None
        self.graph = None
        self.mappings = None

        # # Load CSV paths here
        # self.all_stays = pd.read_csv(config_graph["all_stays_path"])
        # self.all_diagnosis = pd.read_csv(config_graph["all_diagnoses_path"])
        # self.graph_data = {}
        #
        # # load entity dicts
        # self.subjects = {s: i for i, s in enumerate(self.all_stays["SUBJECT_ID"].unique())}  # 33798
        # self.icu_stays = {s: i for i, s in enumerate(self.all_stays["ICUSTAY_ID"].unique())}  # 42276
        # self.diagnosis = {s: i for i, s in enumerate(self.all_diagnosis["ICD9_CODE"].unique())}  # 6169
        #
        # # Load edges
        # self.get_patients_visits()
        # self.get_visits_diagnosis()
        #
        # del self.all_stays
        # del self.all_diagnosis
        # del self.mimic3_ds

        return

    def load_mimic(self):
        # Get mimic dataset from
        raw_path = self.config_graph["mimic3_raw"]
        self.mimic3_ds = MIMIC3Dataset(
            root=raw_path,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
            # tables=["DIAGNOSES_ICD"],
            code_mapping={},
            dev=False,
        )

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

        patients = self.mimic3_ds.patients

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
                    if table == "DIAGNOSES_ICD":
                        for ev in ev_dict["DIAGNOSES_ICD"]:
                            diagnosis_set.add(ev.code)
                            visit_diagnosis_edges.append((ev.visit_id, ev.code))

                    if table == "PROCEDURES_ICD":
                        # Load procedure
                        for ev in ev_dict["PROCEDURES_ICD"]:
                            procedures_set.add(ev.code)
                            visit_procedure_edges.append((ev.visit_id, ev.code))
                    if table == "PRESCRIPTIONS":
                        # Load prescriptions
                        for ev in ev_dict["PRESCRIPTIONS"]:
                            prescriptions_set.add(ev.code)
                            visit_prescription_edges.append((ev.visit_id, ev.code))
                    if table == "LABEVENTS":
                        # Load prescriptions
                        for ev in ev_dict["LABEVENTS"]:
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
        update_edges("visit", "occurs", "labevent", visit_labevent_edges)
        update_edges("visit", "treated", "procedure", visit_procedure_edges)

        # Save mappings
        self.mappings = {
            "patient": patients_dict,
            "visit": visits_dict,
            "diagnosis": diagnosis_dict,
            "procedure": procedures_dict,
            "prescription": prescriptions_dict,
            "labevent": labevents_dict
        }

        with open(f'{self.graph_path}{self.dataset_name}_entity_mapping.pkl', 'wb') as outp:
            pickle.dump(self.mappings, outp, pickle.HIGHEST_PROTOCOL)

        return graph_data

    def set_tasks(self, g):
        mort_pred_samples = self.mimic3_ds.set_task(mortality_prediction_mimic3_fn)
        drug_rec_samples = self.mimic3_ds.set_task(drug_recommendation_mimic3_fn)
        los_samples = self.mimic3_ds.set_task(length_of_stay_prediction_mimic3_fn)
        readm_samples = self.mimic3_ds.set_task(readmission_prediction_mimic3_fn)

        vm = self.mappings["visit"]
        n_nodes = g.num_nodes("visit")
        mort_labels = torch.zeros((n_nodes))

        for s in mort_pred_samples:
            # Assign mortality status
            visit_id = s["visit"]
            mort_labels[visit_id] = s["label"]

        return (
            drug_rec_samples,
            mort_pred_samples,
            los_samples,
            readm_samples
        )

    def get_visits_diagnosis(self):
        subject_stays = self.all_diagnosis[["ICUSTAY_ID", "ICD9_CODE"]].drop_duplicates()
        stays_ids = [self.icu_stays[k] for k in subject_stays["ICUSTAY_ID"]]
        dianoses_ids = [self.diagnosis[k] for k in subject_stays["ICD9_CODE"]]
        self.graph_data.update(
            {
                ("stay", "diagnosed", "diagnosis"): (torch.tensor(stays_ids), torch.tensor(dianoses_ids))
            }
        )

    def get_patients_visits(self):
        subject_stays = self.all_stays[["SUBJECT_ID", "ICUSTAY_ID"]].drop_duplicates()
        subject_ids = [self.subjects[k] for k in subject_stays["SUBJECT_ID"]]
        stays_ids = [self.icu_stays[k] for k in subject_stays["ICUSTAY_ID"]]
        self.graph_data.update(
            {
                ("patient", "visits", "stay"): (torch.tensor(subject_ids), torch.tensor(stays_ids))
            }
        )

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

    @staticmethod
    def set_to_dict(s):
        return {e: i for i, e in enumerate(s)}
