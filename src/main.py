from src.core.application import Application
from src.utils.image_loader import ImageLoader


def main() -> None:
    app = Application()

    image = ImageLoader.load("sample.jpg")

    results = app.run(image)

    print("\n=== EXTRACTOR RESULTS ===")

    for result in results:
        print(result)

    print("\n=== EVIDENCE ===")

    for evidence in app.evidence_store.get_all():
        print(
            f"[{evidence.source}] "
            f"{evidence.evidence_type}: "
            f"{evidence.value!r} "
            f"(confidence={evidence.confidence:.3f})"
        )


if __name__ == "__main__":
    main()
