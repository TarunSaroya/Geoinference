from src.core.application import Application
from src.utils.image_loader import ImageLoader


def main() -> None:
    app = Application()

    image = ImageLoader.load("sample.jpg")

    app.run(image)

    print("\n=== GEOGRAPHIC EVIDENCE ===")

    for evidence in app.evidence_store.get_all():
        print(
            f"[{evidence.source}] "
            f"{evidence.value!r} | "
            f"confidence={evidence.confidence:.3f} | "
            f"relevance={evidence.geographic_relevance} | "
            f"classification="
            f"{evidence.metadata.get('classification', 'unknown')}"
        )
    print("\n=== GEOGRAPHIC METAS ===")

    for meta in app.geographic_meta_store.get_all():
        print(
            f"[{meta.meta_type}] "
            f"{meta.observation} | "
            f"confidence={meta.confidence:.3f} | "
            f"relevance={meta.geographic_relevance} | "
            f"source={meta.source}"
        )

        if meta.metadata:
            for key, value in meta.metadata.items():
                print(f"    {key}: {value}")

        if meta.spatial:
            print(f"    spatial: {meta.spatial}")


if __name__ == "__main__":
    main()
