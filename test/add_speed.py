import xml.etree.ElementTree as ET
import random

# Initialize namespace
namespace = 'http://graphml.graphdrawing.org/xmlns'
ET.register_namespace('', namespace)

# Load the GraphML file
tree = ET.parse('map.graphml')
root = tree.getroot()

# Create and add the new <key> element for speed
key_attribs = {
    'id': 'd22',
    'for': 'edge',
    'attr.name': 'speed',
    'attr.type': 'string'
}
key_element = ET.SubElement(root, 'key', attrib=key_attribs)

# Add speed <data> elements to all <edge> elements
for edge in root.findall('.//edge', {'': namespace}):
    data_attribs = {'key': 'd22'}
    data_element = ET.SubElement(edge, 'data', attrib=data_attribs)
    data_element.text = str(random.randint(2, 8))

# Save changes
tree.write('new_updated_file.graphml')
