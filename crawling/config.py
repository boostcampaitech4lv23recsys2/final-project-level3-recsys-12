class Feature:
    def __init__(self, class_name: str, sementic_tag: str, extract_method: str):
        self.class_name = class_name
        self.sementic_tag = sementic_tag
        self.extract_method = extract_method


class Features:
    def __init__(self):
        self.feature_dict = {
            # item
            "category": Feature(
                class_name="commerce-category-breadcrumb",
                sementic_tag="ol",
                extract_method="special",
            ),
            "rating": Feature(
                class_name="production-selling-header__review__icon",
                sementic_tag="span",
                extract_method="attrs.aria-label",
            ),
            "review": Feature(
                class_name="production-selling-header__review-wrap",
                sementic_tag="p",
                extract_method="text",
            ),
            "price": Feature(
                class_name="production-selling-header__price__original",
                sementic_tag="del",
                extract_method="text",
            ),
            "title": Feature(
                class_name="production-selling-header__title__name",
                sementic_tag="span",
                extract_method="text",
            ),
            "seller": Feature(
                class_name="production-selling-header__title__brand",
                sementic_tag="a",
                extract_method="text",
            ),
            "discount_rate": Feature(
                class_name="production-selling-header__price__discount",
                sementic_tag="span",
                extract_method="text",
            ),
            "image": Feature(
                class_name="production-selling-cover-image__entry__image",
                sementic_tag="img",
                extract_method="attrs.src",
            ),
            "available_product": Feature(
                class_name="production-selling-header__non-selling__text",
                sementic_tag="p",
                extract_method="text",
            ),
            "predict_price": Feature(
                class_name="production-selling-header__non-selling__description",
                sementic_tag="dl",
                extract_method="text",
            ),
            # house
            "detail_table": Feature(
                class_name="project-detail-metadata-detail-item",
                sementic_tag="div",
                extract_method="special",
            ),
            "review_table": Feature(
                class_name="content-detail-stats__item",
                sementic_tag="div",
                extract_method="special",
            ),
            # card
            "img_space": Feature(
                class_name="e59rxfs2", sementic_tag="button", extract_method="text"
            ),
            "img_url": Feature(
                class_name="e11tzz431", sementic_tag="img", extract_method="attrs.src"
            ),
        }


if __name__ == "__main__":
    item = Features()
