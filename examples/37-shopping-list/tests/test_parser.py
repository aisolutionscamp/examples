from shopping_list.parser import Item, parse_item, parse_text, split_items


def test_split_on_commas_newlines_and_and():
    assert split_items("milk, eggs and bread\n- 2 kg potatoes; butter") == [
        "milk", "eggs", "bread", "2 kg potatoes", "butter"]


def test_quantity_and_unit():
    assert parse_item("2 kg potatoes") == Item("potatoes", 2.0, "kg")
    assert parse_item("1,5 l milk") == Item("milk", 1.5, "l")
    assert parse_item("2 cans of tomatoes") == Item("tomatoes", 2.0, "can")


def test_count_without_unit():
    assert parse_item("6 eggs") == Item("eggs", 6.0, None)
    assert parse_item("3x Yogurt") == Item("yogurt", 3.0, None)
    assert parse_item("2 large onions") == Item("large onions", 2.0, None)


def test_plain_name():
    assert parse_text("Bananas") == [Item("bananas")]
