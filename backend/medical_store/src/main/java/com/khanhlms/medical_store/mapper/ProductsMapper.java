package com.khanhlms.medical_store.mapper;

import com.khanhlms.medical_store.dtos.products.requests.CreateProductRequest;
import com.khanhlms.medical_store.dtos.products.requests.IngredientRequest;
import com.khanhlms.medical_store.dtos.products.response.CreateProductResponse;
import com.khanhlms.medical_store.dtos.products.response.DetailProduct;
import com.khanhlms.medical_store.dtos.products.response.IngredientResponse;
import com.khanhlms.medical_store.dtos.products.response.ProductResponse;
import com.khanhlms.medical_store.dtos.questions.response.QuestionResponse;
import com.khanhlms.medical_store.entities.*;
import com.khanhlms.medical_store.exceptions.AppException;
import com.khanhlms.medical_store.exceptions.ErrorCode;
import com.khanhlms.medical_store.repositories.CategoriesRepository;
import com.khanhlms.medical_store.repositories.ManufacturerRepository;
import com.khanhlms.medical_store.utills.CloudinaryUtils;
import org.mapstruct.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.multipart.MultipartFile;

import java.util.Collections;
import java.util.List;
@Mapper(componentModel = "spring")
public abstract class ProductsMapper {
    @Autowired
    protected CloudinaryUtils cloudinaryUtils;
    @Autowired
    protected ManufacturerRepository manufacturerRepository;
    @Autowired
    protected CategoriesRepository categoriesRepository;
    @Autowired
    protected QuestionMapper  questionMapper;

/// /////////////

    @Mapping(source = "images", target = "imageUrl", qualifiedByName = "mapImageUrlPrimary")
    @Mapping(source = "id", target = "id")
    public abstract ProductResponse toProductResponse(ProductsEntity entity);
    @Named("mapImageUrlPrimary")
    protected String mapImageUrlPrimary(List<ImagesEntity> imagesEntities) {
        if (imagesEntities == null || imagesEntities.isEmpty()) {return null;}
        return imagesEntities.get(0).getImageUrl();
    }
///////////////////////////////
    @Mappings({
            @Mapping(source = "category.name", target = "categoryName"),
            @Mapping(source = "manufacturer.name", target = "manufacturerName"),
            @Mapping(source = "images", target = "imageUrl", qualifiedByName = "mapImageUrl"),

    })
    public abstract CreateProductResponse toCreateProductResponse(ProductsEntity productsEntity);

    @Named("mapImageUrl")
    protected List<String> mapImageUrl(List<ImagesEntity> images) {
        if (images == null || images.isEmpty()) {
            return Collections.emptyList();
        }
        return images.stream()
                .map(imagesEntity -> imagesEntity.getImageUrl())
                .toList();
    }
/// //////////////////////
    @Mappings({
            @Mapping(source = "manufacturerId", target = "manufacturer", qualifiedByName = "mapManufacturerById"),
            @Mapping(source = "categoryId", target = "category", qualifiedByName = "mapCategoryById"),
            @Mapping(source = "images", target = "images", qualifiedByName = "mapImagesEntity"),
            @Mapping(source = "price", target = "originPrice"),
            @Mapping(source = "currency", target = "currency"),
            @Mapping(source = "quantity", target = "quantity")
    })
    public abstract ProductsEntity toEntity(CreateProductRequest request);

    @Named("mapManufacturerById")
    protected ManufacturerEntity mapManufacturerById(String manufacturerId) {
        if(manufacturerId == null) {return null;}
        ManufacturerEntity manufacturer = manufacturerRepository.findById(manufacturerId)
                .orElseThrow(() -> new AppException(ErrorCode.MANUFACTURER_NOT_FOUND));
        return manufacturer;
    }
    @Named("mapCategoryById")
    protected CategoryEntity  mapCategoryById(String categoryId) {
        if (categoryId == null) {return null;}
        CategoryEntity category = this.categoriesRepository.findById(categoryId)
                .orElseThrow(()-> new AppException(ErrorCode.CATEGORY_NOT_FOUND));
        return category;
    }
    @Named("mapImagesEntity")
    protected List<ImagesEntity> mapImagesEntity(List<MultipartFile> images) {
        if(images == null) {return null;}
        return images.stream()
                .map(image -> {
                    String imageUrl = cloudinaryUtils.uploadImageCloddy(image);
                    ImagesEntity entity = ImagesEntity.builder()
                            .imageUrl(imageUrl)
                            .build();
                    return entity;
                })
                .toList();
    }
    /// ///////////////////////////////////
    @Mappings({
            @Mapping(source = "manufacturer.name", target = "manufacturer"),
            @Mapping(source = "category.name", target = "category"),
            @Mapping(source = "images", target = "images", qualifiedByName = "mapImageUrl"),
            @Mapping(source = "questions", target = "questions", qualifiedByName = "mapQuestions"),
            @Mapping(source = "manufacturer.id", target = "manufactureId")
    })
    public abstract DetailProduct toDetailProduct(ProductsEntity entity);
    @Named("mapQuestions")
    protected List<QuestionResponse> mapQuestions(List<QuestionEntity> questions) {
        if (questions == null) {return Collections.emptyList();}
        return questions.stream()
                .map(questionEntity -> this.questionMapper.toQuestionResponse(questionEntity))
                .toList();
    }
    @Named("mapIngredientToResponse")
    protected List<IngredientResponse> mapIngredientToResponse(List<IngredientEntity> ingredients) {
        if (ingredients == null) {return Collections.emptyList();}
        return ingredients.stream()
                .map(ingredientEntity -> {
                    return IngredientResponse.builder()
                            .name(ingredientEntity.getName())
                            .description(ingredientEntity.getDescription())
                            .unit(ingredientEntity.getUnit())
                            .amount(ingredientEntity.getAmount())
                            .build();
                })
                .toList();
    }
}
