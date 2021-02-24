import numpy as np

def sort_popular(data, top_item_quantity):
    grouped_by_quantity_data = data.groupby('item_id')['quantity'].count()
    grouped_by_quantity_without_indexes_data = grouped_by_quantity_data.reset_index()
    grouped_by_quantity_without_indexes_data.rename(columns={'quantity': 'n_sold'}, inplace=True)
    
    return grouped_by_quantity_without_indexes_data.sort_values('n_sold', ascending=False).\
                                        head(top_item_quantity).item_id.tolist()

def prefilter_items(data, item_features, top_item_quantity=5000, first_sort_popular=True):
    if first_sort_popular:
        item_quantity_top = sort_popular(data, top_item_quantity)
        data.loc[~data['item_id'].isin(item_quantity_top), 'item_id'] = 999999
        
    # Уберем самые популярные 
    popularity = (data.groupby('item_id')['user_id'].nunique() / data.user_id.nunique()).reset_index() 
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_popular = popularity[popularity.share_unique_users > 0.2].item_id.tolist()
    data.loc[data['item_id'].isin(top_popular), 'item_id'] = 999999
    
    # Уберем самые непопулярные 
    unpopular = popularity[popularity.share_unique_users < 0.02].item_id.tolist()
    data.loc[data['item_id'].isin(unpopular), 'item_id'] = 999999 
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    weeks_in_year = 12
    time = data.week_no.max() - weeks_in_year
    sold_last_12_month = data[data.week_no >= time].item_id.unique().tolist()
    data.loc[~data['item_id'].isin(sold_last_12_month), 'item_id'] = 999999
    
    # Уберем не интересные для рекоммендаций категории (department)
    department_size = item_features.groupby('department')['item_id'].nunique().\
                                       sort_values(ascending=False).reset_index()
    department_size.rename(columns={'item_id': 'n_items'}, inplace=True)
    departments = department_size[department_size.n_items < 150].department.tolist()

    items_in_deparmments = item_features[
        item_features.department.isin(departments)
    ].item_id.unique().tolist()
    data.loc[data['item_id'].isin(items_in_deparmments), 'item_id'] = 999999
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / np.maximum(data['quantity'], 1)
    data.loc[data['price'] < 1, 'item_id'] = 999999
    
    # Уберем слишком дорогие товары
    data.loc[data['price'] > 50, 'item_id'] = 999999
    
    if not first_sort_popular:
        # Оставим только n самых популярных товаров после фильтрации
        popularity = data.groupby('item_id')['quantity'].count().reset_index() #### sum
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_top_item_quantity = popularity.sort_values('n_sold', ascending=False).\
                                        head(top_item_quantity).item_id.tolist()
        #добавим, чтобы не потерять пользователей
        data.loc[~data['item_id'].isin(top_top_item_quantity), 'item_id'] = 999999
        
    return data
