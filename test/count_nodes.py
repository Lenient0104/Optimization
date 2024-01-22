import xml.etree.ElementTree as ET


def count_unique_node_ids(file_name):
    # Parse the XML file
    try:
        tree = ET.parse(file_name)
    except Exception as e:
        print(f"Error while parsing the file: {e}")
        return

    # Get the root of the XML document
    root = tree.getroot()

    # Use a set to store unique node ids
    unique_ids = set()

    # Try to find all 'node' tags in the XML and add their ids to the set
    try:
        for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
            # Get the id attribute of the 'node' tag
            node_id = node.get('id')

            # Add the id to the set of unique ids
            unique_ids.add(node_id)
    except Exception as e:
        print(f"Error while processing nodes: {e}")
        return

    # The number of unique node ids is the size of the set
    return len(unique_ids)


# Replace 'filename.graphml' with the actual name of your file
file_name = 'map.graphml'
print("Number of unique node ids in the file: ", count_unique_node_ids(file_name))
