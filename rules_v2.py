import cv2
import rules


def rule1_metadata_v2(target, input_path):
    out = rules.rule1_metadata(target, input_path)
    out["out_of"] = 25
    out["score"] = round(out["score"] * 25 / 30)
    return out


def rule2_histogram_v2(target, input_path):
    out = rules.rule2_histogram(target, input_path)
    out["out_of"] = 25
    out["score"] = round(out["score"] * 25 / 30)
    return out


def rule3_template_v2(target, input_path):
    return rules.rule3_template(target, input_path)


RULES = (
    "rule1_metadata_v2",
    "rule2_histogram_v2",
    "rule3_template_v2",
    "rule4_edges",
)


def rule4_edges(target, input_path):
    out = {
        "rule": 4,
        "name": "Edges",
        "fired": False,
        "score": 0,
        "out_of": 10,
        "note": "Edge similarity 0.00",
        "metric": 0.0,
    }

    try:
        src = cv2.imread(target["path"], cv2.IMREAD_GRAYSCALE)
        suspect = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if src is None or suspect is None:
            return out

        src = cv2.resize(src, (256, 256))
        suspect = cv2.resize(suspect, (256, 256))

        src_edges = cv2.Canny(src, 100, 200)
        suspect_edges = cv2.Canny(suspect, 100, 200)

        difference = cv2.norm(
            src_edges,
            suspect_edges,
            cv2.NORM_L1
        )

        similarity = 1.0 - (
            difference / (255.0 * 256 * 256)
        )

        similarity = max(0.0, min(1.0, similarity))

        out["metric"] = round(similarity, 3)
        out["note"] = f"Edge similarity {out['metric']:.2f}"

        if similarity >= 0.5:
            out["fired"] = True
            out["score"] = round(10 * similarity)

    except Exception:
        pass

    return out
