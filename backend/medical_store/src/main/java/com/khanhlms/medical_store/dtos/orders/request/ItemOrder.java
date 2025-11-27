package com.khanhlms.medical_store.dtos.orders.request;

import lombok.*;
import lombok.experimental.FieldDefaults;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class ItemOrder {
    String productId;
    Integer quantity;
}
