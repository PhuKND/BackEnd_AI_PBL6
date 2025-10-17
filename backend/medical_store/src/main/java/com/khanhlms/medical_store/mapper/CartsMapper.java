package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.cartItems.response.CartItemResponse;
import com.khanhlms.medical_store.entities.CartItemEntity;
import com.khanhlms.medical_store.entities.DiscountEntity;
import com.khanhlms.medical_store.entities.ImagesEntity;
import com.khanhlms.medical_store.repositories.ProductRepository;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;
import org.mapstruct.Named;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;

@Mapper(componentModel = "spring")
public abstract class CartsMapper {
    @Autowired
    protected ProductRepository productRepository;

    // ///////////////////////////////////
    @Mappings({
            @Mapping(source = "product.id", target = "productId"),
            @Mapping(source = "product.name", target = "productName" ),
            @Mapping(source = "product.images", target = "imageUrl", qualifiedByName = "toImageURL"),
            @Mapping(source = ".", target = "totalPrice", qualifiedByName = "calculateTotalPrice"),
            @Mapping(source = "product.originPrice", target = "originPrice"),
            @Mapping(source = "product.discount.message", target = "messageDiscount" ),
            @Mapping(source = "product.discount.startDate", target = "discountStart" ),
            @Mapping(source = "product.discount.endDate", target = "discountEnd"),
            @Mapping(source = "product.discount.percent", target = "percent"),
            @Mapping(source = "product.unit", target = "unit"),
            @Mapping(source = "product.currency", target = "currency")
    })
    public abstract CartItemResponse toCartItemResponse(CartItemEntity cartItemEntity);
    @Named("toImageURL")
    protected String toImageURL(List<ImagesEntity> images) {
        if (images == null || images.isEmpty()) {
            return null;
        }
        return images.get(0).getImageUrl();
    }
    @Named("calculateTotalPrice")
    protected Double calculateTotalPrice(CartItemEntity cartItem) {
        if (cartItem == null || cartItem.getProduct() == null) return 0.0;

        DiscountEntity discount = cartItem.getProduct().getDiscount();
        double percent = (discount != null && discount.getPercent() != null)
                ? discount.getPercent()
                : 0.0;

        return cartItem.getProduct().getOriginPrice() * cartItem.getQuantity() * (1 - percent / 100);
    }

}
