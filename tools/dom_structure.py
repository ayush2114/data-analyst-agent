from bs4 import BeautifulSoup, Comment

def extract_dom_structure_with_identifiers(html, max_depth=6):
    soup = BeautifulSoup(html, "html.parser")
    
    def format_tag(tag):
        parts = [tag.name]
        if tag.get("id"):
            parts.append(f"#{tag['id']}")
        if tag.get("class"):
            class_str = "." + ".".join(tag.get("class"))
            parts.append(class_str)
        return "".join(parts)

    def traverse(node, depth=0):
        if depth > max_depth:
            return []
        lines = []
        for child in node.children:
            if isinstance(child, Comment) or child.name in ['script', 'style']:
                continue
            if hasattr(child, 'name') and child.name:
                lines.append("  " * depth + format_tag(child))
                lines.extend(traverse(child, depth + 1))
        return lines

    return "\n".join(traverse(soup))

# Usage
with open("scraped_content.html", encoding="utf-8") as f:
    html = f.read()

# print(extract_dom_structure_with_identifiers(html))