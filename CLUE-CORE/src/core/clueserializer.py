import ast
import os
import xml.etree.ElementTree as ET
from core.cluerun import ClueRun, ClueRound, ClueConfig

def clueconfig_to_xml(config):
    elem = ET.Element("ClueConfig")
    for attr in [
        "algorithm", "distanceMetric", "paramOptimization", "epsilon", "minPts",
        "hyperplanes", "hashtables", "kClusters", "threads", "standardize",
        "useFeatures"
    ]:
        value = getattr(config, attr)
        ET.SubElement(elem, attr).text = str(value)
    return elem

def clueconfig_from_xml(elem):
    kwargs = {}
    for attr in [
        "algorithm", "distanceMetric", "paramOptimization", "epsilon", "minPts",
        "hyperplanes", "hashtables", "kClusters", "threads", "standardize",
        "useFeatures"
    ]:
        text = elem.find(attr).text
        # Convert types appropriately
        if attr in ["algorithm", "distanceMetric", "paramOptimization", "minPts", "hyperplanes", "hashtables", "kClusters", "threads"]:
            kwargs[attr] = int(text)
        elif attr in ["epsilon"]:
            kwargs[attr] = float(text)
        elif attr in ["standardize", "useFeatures"]:
            kwargs[attr] = text == "True"
        else:
            kwargs[attr] = text
    return ClueConfig(**kwargs)

def clueround_to_xml(round_obj):
    elem = ET.Element("ClueRound")
    ET.SubElement(elem, "roundName").text = str(round_obj.roundName)
    try:
        directory = os.path.relpath(round_obj.directory)
        ET.SubElement(elem, "directory").text = directory
    except ValueError:
        ET.SubElement(elem, "directory").text = str(round_obj.directory)
    ET.SubElement(elem, "featureSelection").text = str(round_obj.featureSelection)
    ET.SubElement(elem, "clusterSelection").text = str(round_obj.clusterSelection)
    elem.append(clueconfig_to_xml(round_obj.clueConfig))
    return elem

def clueround_from_xml(elem):
    roundName = elem.find("roundName").text
    directory = elem.find("directory").text
    clusterSelectionText = elem.find("clusterSelection").text
    featureSelectionText = elem.find("featureSelection").text

    try: 
        if clusterSelectionText and clusterSelectionText != "None":
            clusterSelection = ast.literal_eval(clusterSelectionText)
        else:
            clusterSelection = None
    except (ValueError, SyntaxError):
        clusterSelection = None
    try: 
        if featureSelectionText and featureSelectionText != "None":
            featureSelection = ast.literal_eval(featureSelectionText)
        else:
            featureSelection = None
    except (ValueError, SyntaxError):
        featureSelection = None
    config_elem = elem.find("ClueConfig")
    clueConfig = clueconfig_from_xml(config_elem)
    return ClueRound(roundName, directory, featureSelection, clusterSelection, clueConfig)

def cluerun_to_xml(run_obj):
    elem = ET.Element("ClueRun")
    ET.SubElement(elem, "runName").text = str(run_obj.runName)
    try:
        baseFileRP = os.path.relpath(run_obj.baseFile)
        ET.SubElement(elem, "baseFile").text = baseFileRP
    except ValueError:
        ET.SubElement(elem, "baseFile").text = str(run_obj.baseFile)
    try:
        baseDirectoryRP = os.path.relpath(run_obj.baseDirectory)
        ET.SubElement(elem, "baseDirectory").text = baseDirectoryRP
    except ValueError:
        ET.SubElement(elem, "baseDirectory").text = str(run_obj.baseDirectory)
    try:
        CLUECLUSTRP = os.path.relpath(run_obj.CLUECLUST)
        ET.SubElement(elem, "CLUECLUST").text = CLUECLUSTRP
    except ValueError:
        ET.SubElement(elem, "CLUECLUST").text = str(run_obj.CLUECLUST)
    ET.SubElement(elem, "outputDirectory").text = str(run_obj.outputDirectory)
    ET.SubElement(elem, "interactive").text = str(run_obj.interactive)
    rounds_elem = ET.SubElement(elem, "Rounds")
    for round_obj in run_obj.rounds:
        rounds_elem.append(clueround_to_xml(round_obj))
    return elem

def cluerun_from_xml(elem):
    runName = elem.find("runName").text
    baseFile = elem.find("baseFile").text
    baseDirectory = elem.find("baseDirectory").text
    outputDirectory = elem.find("outputDirectory").text
    interactive = elem.find("interactive").text == "True"
    CLUECLUST = elem.find("CLUECLUST").text
    rounds = []
    for round_elem in elem.find("Rounds").findall("ClueRound"):
        rounds.append(clueround_from_xml(round_elem))
    run = ClueRun(runName, baseFile, baseDirectory, outputDirectory, interactive, CLUECLUST)
    run.rounds = rounds
    return run

def serialize_cluerun(run_obj, file_path):
    # Pretty print the XML
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                indent(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    root = cluerun_to_xml(run_obj)
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)

def deserialize_cluerun(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return cluerun_from_xml(root)