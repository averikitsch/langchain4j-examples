
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.engine.EmbeddingStoreConfig;
import dev.langchain4j.engine.AlloyDBEngine;
import dev.langchain4j.engine.MetadataColumn;
import dev.langchain4j.store.embedding.alloydb.AlloyDBEmbeddingStore;
import java.sql.SQLException;


import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class AlloyDBEmbeddingStoreWithMetadataExample {

    private static final String TABLE_NAME = "EMBEDDING_TEST_TABLE";
    private static final Integer VECTOR_SIZE = 384;

    public static void main(String[] args) throws SQLException {

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        AlloyDBEngine engine = new AlloyDBEngine.Builder()
                .projectId(System.getenv("ALLOYDB_PROJECT_ID"))
                .region(System.getenv("ALLOYDB_REGION"))
                .cluster(System.getenv("ALLOYDB_CLUSTER"))
                .instance(System.getenv("ALLOYDB_INSTANCE"))
                .database(System.getenv("ALLOYDB_DB_NAME"))
                .user(System.getenv("ALLOYDB_USER"))
                .password(System.getenv("ALLOYDB_PASSWORD"))
                .ipType("public")
                .build();

        engine.getConnection().createStatement().executeUpdate(String.format(
            "DROP TABLE IF EXISTS \"%s\"", TABLE_NAME));
                
        List<MetadataColumn> metadataColumns = new ArrayList<>();
        metadataColumns.add(new MetadataColumn("userId", "uuid", true));

        EmbeddingStoreConfig embeddingStoreConfig = new EmbeddingStoreConfig.Builder(TABLE_NAME, VECTOR_SIZE)
                .metadataColumns(metadataColumns)
                .storeMetadata(true)
                .build();

        engine.initVectorStoreTable(embeddingStoreConfig);

        List<String> metaColumnNames =
                metadataColumns.stream().map(c -> c.getName()).collect(Collectors.toList());

        AlloyDBEmbeddingStore embeddingStore = new AlloyDBEmbeddingStore.Builder(engine, TABLE_NAME)
                .metadataColumns(metaColumnNames)
                .build();


        Metadata metadata1 = new Metadata();
        metadata1.put("userId", UUID.randomUUID());
        TextSegment segment1 = TextSegment.from("I like turtles.", metadata1);
        Embedding embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);

        Metadata metadata2 = new Metadata(); 
        metadata2.put("userId", UUID.randomUUID());
        TextSegment segment2 = TextSegment.from("All right!. You're a great zombie", metadata2);
        Embedding embedding2 = embeddingModel.embed(segment2).content();
        embeddingStore.add(embedding2, segment2);

        Embedding queryEmbedding = embeddingModel.embed("What is your favourite animal?").content();
        EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(1)
                .build();
        EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);

        EmbeddingMatch<TextSegment> embeddingMatch = searchResult.matches().get(0);
        System.out.println(embeddingMatch.score());
        System.out.println(embeddingMatch.embedded().text());
    }
}
